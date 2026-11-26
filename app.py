"""
STEP 5: The website
-------------------
Run:  streamlit run app.py
Then your browser opens at http://localhost:8501

All the detection logic lives in detector.py. This file only builds the page.
"""

import pandas as pd
import streamlit as st

from detector import PhishingDetector

st.set_page_config(page_title="Phishing Website Detector", page_icon="🛡️")


@st.cache_resource  # load the AI model only once, not on every click
def load_detector():
    return PhishingDetector()


detector = load_detector()

if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("About this system")
    st.write(
        "This system uses a machine learning model (**Linear SVM**) trained on "
        "251,941 website domains from two datasets: the PhiUSIIL Phishing URL "
        "Dataset and a newer phishing URL dataset."
    )
    st.markdown(
        "**Test results**\n"
        "- Accuracy: 88.4%\n"
        "- Precision: 93.3%\n"
        "- Recall: 79.9%\n"
        "- F1 score: 0.861"
    )
    st.subheader("How it works")
    st.markdown(
        "1. The link is cleaned down to its domain name.\n"
        "2. Shared platforms and link shorteners (Google Docs, bit.ly...) "
        "are flagged for a careful manual check.\n"
        "3. Well-known safe sites are checked against a trusted list.\n"
        "4. Other domains are analysed by the AI model.\n"
        "5. An adult content filter blocks known adult websites.\n"
        "6. Warning signs are listed to explain the result."
    )
    st.subheader("Try these")
    st.code("shopee.ph\nsecure-bdo-online.weebly.com\ngcash-rewards.web.app/claim\n"
            "bit.ly/3xYz\n"
            "paypal-verify-account.xyz\nuni-mainz.de", language=None)

# ----------------------------------------------------------------- Main page
st.title("🛡️ AI-Based Phishing Website Detector")
st.write("Paste a link below to check if it is a phishing website.")

with st.form("check_form"):
    url = st.text_input("Website link", placeholder="e.g. https://example.com/login")
    submitted = st.form_submit_button("Check link")

if submitted:
    if not url.strip():
        st.warning("Please paste a link first.")
    else:
        result = detector.analyze_url(url)

        if not result["valid"]:
            st.warning("That doesn't look like a website link. "
                       "Try something like **example.com**.")
        else:
            verdict = result["verdict"]
            prob = result["probability"]

            # --- Verdict box
            if verdict == "Phishing":
                st.error(f"### 🚨 Phishing\nThis website is very likely a scam. "
                         f"Do not enter passwords, OTPs, or payment details.")
            elif verdict == "Adult content":
                st.warning("### 🔞 Adult content: blocked\nThis website contains "
                           "adult (18+) content. It was blocked by the adult "
                           "content filter. This is not a phishing result.")
            elif verdict == "Check carefully":
                st.info("### 🔍 Check carefully\nThis link is on a shared platform "
                        "or link shortener. Anyone can post pages here, so the "
                        "website name alone can't tell if it's safe. Check who "
                        "sent it and never enter passwords or OTPs from a link "
                        "like this.")
            elif verdict == "Suspicious":
                st.warning(f"### ⚠️ Suspicious\nThis website shows signs of "
                           f"phishing. Be careful and double-check the link.")
            else:
                st.success(f"### ✅ {verdict}\nNo strong signs of phishing "
                           f"were found.")

            col1, col2 = st.columns(2)
            col1.metric("Domain checked", result["domain"])
            if result["source"] == "Trusted list":
                col2.metric("Checked by", "Trusted list")
                st.caption("This is a well-known official website, so the AI "
                           "model was not needed.")
            elif result["source"] == "Adult content filter":
                col2.metric("Checked by", "Adult content filter")
            elif result["source"] == "Shared platform":
                col2.metric("Checked by", "Shared platform")
                st.caption("The AI reads website names only. On shared platforms "
                           "every page has the same name, so the AI is not used.")
            else:
                col2.metric("Phishing probability", f"{prob:.1%}")
                st.progress(prob)

            # --- Explanation
            if result["signs"]:
                st.subheader("Warning signs found")
                for sign in result["signs"]:
                    st.write(f"• {sign}")

            if result["source"] == "AI model":
                with st.expander("What did the AI look at?"):
                    st.write(
                        "The AI reads the domain as small pieces of 2-5 characters. "
                        "These pieces had the biggest influence on this result:"
                    )
                    left, right = st.columns(2)
                    with left:
                        st.markdown("**🔴 Pushed toward phishing**")
                        for piece, _ in result["phishing_pieces"] or [("(none)", 0)]:
                            st.code(piece, language=None)
                    with right:
                        st.markdown("**🟢 Pushed toward legitimate**")
                        for piece, _ in result["legit_pieces"] or [("(none)", 0)]:
                            st.code(piece, language=None)

            st.session_state.history.insert(0, {
                "Link": url,
                "Domain": result["domain"],
                "Result": verdict,
                "Phishing probability": ("-" if result["source"] != "AI model"
                                         else f"{prob:.1%}"),
            })

# ----------------------------------------------------------------- History
if st.session_state.history:
    st.divider()
    st.subheader("Recent checks")
    st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True)

st.divider()
st.caption(
    "⚠️ This is a student prototype. The AI judges the website name only and "
    "can make mistakes (about 1 in 5 phishing sites, mostly ones with "
    "normal-looking names, may be missed). Always be careful with links you receive."
)