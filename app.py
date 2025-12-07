# IMPORT LIBRARIES----------------------
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import difflib

# SESSION STATE -------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# BACKGROUND IMAGE -------------------------
st.markdown(
        f"""
        <style>
        .stApp {{
        background-image: url("https://i.imgur.com/NiiMIOk.jpeg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        }}
        </style>""",
        unsafe_allow_html=True
    )

# LOAD MODELS / DATA ------------------------- 
@st.cache_data
def load_apriori_results():
    frequent_itemset= joblib.load('frequent_itemset.joblib')
    rules= joblib.load('rules.joblib')
    return frequent_itemset, rules
 
@st.cache_data
def load_dataset():
    df= pd.read_csv("Groceries_dataset.csv")
    df['itemDescription']= df['itemDescription'].str.strip()

    basket= df.groupby(["Member_number", "Date"])["itemDescription"].apply(list).reset_index()
    # ONE HOT ENCODING 
    all_items= sorted(df["itemDescription"].unique())
    for item in all_items:
        basket[item]= basket["itemDescription"].apply(lambda x:1 if item in x else 0)

    basket_ohe =basket[all_items]
    return df, basket, basket_ohe, all_items

@st.cache_data
def load_chatbot_corpus():
    return joblib.load("chatbot_corpus.joblib")
     
@st.cache_data
def load_credentials():
    return joblib.load("creds.joblib")
   
# LOAD ALL ASSETS ---------------------------
df, basket, basket_ohe, all_items= load_dataset()
frequent_itemsets, rules= load_apriori_results()
corpus= load_chatbot_corpus()
credentials= load_credentials()

# STREAMLIT UI----------------------- 
st.title("🛒 Market Basket Analysis Aplication")
tab1, tab2, tab3= st.tabs(["📝 Dataset Info", "🤖 Chatbot", "🔐 Login & Analysis"])

# tab1 ----------
with tab1:
    st.markdown(
        """ 
        <h2 style='text-algin: center;'> 🔍 Apriori Algorithm Dashboard</h2>
        <p style='text-align:center; font-size:18px;'>
            This application demonstrates the <b>Apriori Algorithm</b> used for 
            discovering frequent itemsets and association rules in transactional data.
            <br><br>
            Explore the dataset, generate insights, and analyze relationships 
            between grocery items.
            Click below to explore dataset details.
        </p>
        """, 
        unsafe_allow_html= True
    )
    st.write("---")

# EXPANDER -------------------
    with st.expander("📝 View Dataset Info"):
        st.subheader("📄 Preview")
        st.dataframe(df.head())

        st.subheader("📏 Dataset shape")
        st.write(f"Rows: {df.shape[0]}, columns: {df.shape[1]}")

        st.subheader("📊 Statistics Summary")
        st.write(df.describe())

        st.subheader("Top 10 Most Frequent Items")
        item_count= df['itemDescription'].value_counts().head(10)

        fig, ax= plt.subplots()
        fig.patch.set_alpha(0.0)
        ax.set_facecolor('none')
        bars= ax.bar(item_count.index, item_count.values)
        for bar in bars:
            ax.text(
                bar.get_x()+ bar.get_width()/2,
                bar.get_height(),
                str(int(bar.get_height())),
                ha='center',
                va= 'bottom'
            )
        plt.xticks(rotation=45)
        plt.xlabel("Items")
        plt.ylabel("Frequency")
        plt.title("Top 10 Most Frequent Items")
        st.pyplot(fig)

# TAB2 ----------------------------
with tab2:
    st.header("🤖 Chatbot ")
    user_query= st.text_input("Ask something:")
    if user_query:
        keys= list(corpus.keys())
        match= difflib.get_close_matches(user_query.lower(), keys, n=1, cutoff=0.4)
        if match:
            st.success(corpus[match[0]])
        else:
            st.warning("Sorry, I did not understand this question")

# TAB3 ------------------------
with tab3:
    st.header("🔐 Login to Access Analysis Tools")
    
    # LOGIN-----------------
    if not st.session_state.logged_in:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username in credentials and credentials[username] == password:
                st.session_state.logged_in = True
                st.success("Login Successful!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    # ANALYSIS TOOLS -----------------------
    else:
        st.success("You are logged in ✔️")
        t1, t2 = st.tabs(["📊 Support & Confidence Analysis", "🔍 Item Pair Probability Checker"])

        with t1:
            st.header("📊 Precomputed Apriori Results")

            min_support = st.slider("Minimum Support", 0.0, 1.0, 0.1, 0.01)
            min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.1, 0.01)

            filtered_itemsets = frequent_itemsets[frequent_itemsets["support"] >= min_support]
            filtered_rules = rules[
                (rules["support"] >= min_support) &
                (rules["confidence"] >= min_confidence)
            ]

            st.subheader("Frequent Itemsets")
            st.dataframe(filtered_itemsets)

            st.subheader("Association Rules")
            st.dataframe(filtered_rules)

        with t2:
            st.header("🔍 Item Pair Probability Checker")

            item1 = st.selectbox("Select Item 1", all_items, index=None, key="item1")
            item2 = st.selectbox("Select Item 2", all_items, index=None, key="item2")

            if st.button("Calculate Association", key="calc_btn"):
                if item1 is None or item2 is None:
                    st.warning("Please select items.")
                    st.stop()
                if item1 == item2:
                    st.warning("Please choose two different items.")
                    st.stop()

                total = len(basket)
                # SUPPORT --------------------
                s1 = basket_ohe[item1].sum() / total
                s2 = basket_ohe[item2].sum() / total
                s12 = ((basket_ohe[item1] & basket_ohe[item2]).sum()) / total
                # CONFIDENCE -----------------
                conf_1_to_2 = s12 / s1 if s1 > 0 else 0
                conf_2_to_1 = s12 / s2 if s2 > 0 else 0
                # LIFT --------------------------------
                lift_1_to_2 = conf_1_to_2 / s2 if s2 > 0 else 0

                st.subheader("📈 Calculated Metrics")
                st.write(f"**Support({item1}) = {s1:.4f}**")
                st.write(f"**Support({item2}) = {s2:.4f}**")
                st.write(f"**Support({item1}, {item2}) = {s12:.4f}**")
                st.write("---")
                st.write(f"**Confidence({item1} → {item2}) = {conf_1_to_2:.4f}**")
                st.write(f"**Confidence({item2} → {item1}) = {conf_2_to_1:.4f}**")
                st.write("---")
                st.write(f"**Lift({item1} → {item2}) = {lift_1_to_2:.4f}**")

                st.subheader("Interpretation")
                if lift_1_to_2 > 1:
                    st.success("These items are Positively Associated (often bought together).")
                elif lift_1_to_2 == 1:
                    st.info("These items are independent.")
                else:
                    st.warning("These items are negatively associated.")
