"""
detector.py - the "brain" of the system
---------------------------------------
The website (app.py) calls analyze_url() from here. Keeping the logic
separate from the website makes it easier to test and to explain.

How a URL is checked:
  1. Clean it into a bare domain        (same extract_domain as Step 2)
  2. Is it a SHARED platform/shortener? -> "Check carefully", AI is skipped
  3. Is it on the TRUSTED list?         -> Legitimate, AI is skipped
  4. Otherwise, ask the AI model        -> probability of phishing
  4. Explain the result:
       - which character pieces pushed the AI toward phishing/legitimate
       - simple warning signs a human can understand (rule-based)
"""

import ipaddress
from urllib.parse import urlparse

import joblib
import numpy as np

MODEL_PATH = "model/phishing_model.joblib"

# -----------------------------------------------------------------------------
# Adult content filter (separate from phishing - adult sites are NOT phishing,
# so they are NOT added to the AI's training data).
# -----------------------------------------------------------------------------
# 1. Known adult websites (subdomains count too). Add more as needed.
ADULT_DOMAINS = {
    "pornhub.com", "xvideos.com", "xnxx.com", "xhamster.com", "redtube.com",
    "youporn.com", "tube8.com", "spankbang.com", "eporner.com", "beeg.com",
    "youjizz.com", "tnaflix.com", "porntrex.com", "hqporner.com", "txxx.com",
    "chaturbate.com", "stripchat.com", "bongacams.com", "livejasmin.com",
    "cam4.com", "camsoda.com", "myfreecams.com", "onlyfans.com", "fansly.com",
    "brazzers.com", "realitykings.com", "bangbros.com", "nhentai.net",
    "hanime.tv", "rule34.xxx", "e-hentai.org", "motherless.com", "xxxbunker.com"
    "pinaypie.com",
}
# 2. Domain endings created specifically for adult content.
ADULT_TLDS = {"xxx", "porn", "adult", "sex"}
# 3. Keywords. ONLY words we tested against our 251,941 domains and found to
#    match no normal sites. Words like "sex" (middlesex.ca), "anal"
#    (canalsocietynj.org), "cock" (hancockhouse.com) or "xvideo"
#    (jeuxvideo.com, a gaming site) were rejected because they block
#    innocent websites.
ADULT_KEYWORDS = {"porn", "xxx", "hentai", "xnxx", "xhamster", "xvideos",
                  "nsfw", "camgirl"}
# 4. Exceptions: normal sites that happen to contain an adult keyword.
#    If the filter wrongly blocks a site, add it here.
ADULT_EXCEPTIONS = {"pornokitsch.com"}  # a book/genre-fiction review blog


def adult_check(domain):
    """Return the reason a domain is adult content, or None."""
    if any(domain == d or domain.endswith("." + d) for d in ADULT_EXCEPTIONS):
        return None
    if any(domain == d or domain.endswith("." + d) for d in ADULT_DOMAINS):
        return "This is a known adult website."
    tld = domain.rsplit(".", 1)[-1]
    if tld in ADULT_TLDS:
        return f"Ends in '.{tld}', a domain ending made for adult content."
    words = sorted(k for k in ADULT_KEYWORDS if k in domain)
    if words:
        return "Website name contains adult keywords: " + ", ".join(words) + "."
    return None


# Risk levels shown to the user (the model's decision line is 50%,
# the same threshold used in Steps 3 and 4).
PHISHING_THRESHOLD = 0.70
SUSPICIOUS_THRESHOLD = 0.50

# -----------------------------------------------------------------------------
# Trusted domains: well-known sites that are checked BEFORE the AI.
# Real security tools also combine an allow-list with machine learning.
# Subdomains count too (m.facebook.com -> facebook.com).
# Add your own as needed.
# -----------------------------------------------------------------------------
TRUSTED_DOMAINS = {
    # Global
    "google.com", "youtube.com", "facebook.com", "instagram.com", "messenger.com",
    "whatsapp.com", "x.com", "twitter.com", "tiktok.com", "linkedin.com",
    "microsoft.com", "live.com", "outlook.com", "office.com", "apple.com",
    "icloud.com", "amazon.com", "netflix.com", "spotify.com", "github.com",
    "wikipedia.org", "yahoo.com", "reddit.com", "paypal.com", "zoom.us",
    "canva.com", "openai.com", "chatgpt.com", "anthropic.com", "claude.ai",
    # Philippines
    "gcash.com", "paymaya.com", "maya.ph", "bdo.com.ph", "bpi.com.ph",
    "metrobank.com.ph", "landbank.com", "pnb.com.ph", "securitybank.com",
    "unionbankph.com", "rcbc.com", "chinabank.ph", "shopee.ph", "lazada.com.ph",
    "grab.com", "foodpanda.ph", "globe.com.ph", "smart.com.ph", "pldthome.com",
    "gov.ph", "sss.gov.ph", "philhealth.gov.ph", "pagibigfund.gov.ph",
    "bir.gov.ph", "dfa.gov.ph", "lto.gov.ph", "deped.gov.ph", "ched.gov.ph",
}

# Free website builders / hosting where ANYONE can create a page.
# Even if the main platform is trusted (e.g. google.com), pages here are not.
USER_HOSTING_PLATFORMS = {
    "sites.google.com", "docs.google.com", "forms.gle", "firebaseapp.com",
    "web.app", "weebly.com", "weeblysite.com", "wixsite.com", "godaddysites.com",
    "000webhostapp.com", "blogspot.com", "github.io", "netlify.app",
    "vercel.app", "pages.dev", "glitch.me", "repl.co", "ipfs.io",
}

# Shared platforms and link shorteners where the website name is the SAME
# for safe and scam pages (docs.google.com/... can be a class handout or a
# fake login form; bit.ly/... can lead anywhere). The AI only reads the name,
# so it cannot judge these. They are removed from training (prepare_data.py)
# and get a special "check carefully" result on the website.
SHARED_HOSTS = {
    # File / page sharing
    "docs.google.com", "sites.google.com", "drive.google.com", "forms.gle",
    "storage.googleapis.com", "firebasestorage.googleapis.com",
    "dropbox.com", "onedrive.live.com", "1drv.ms",
    "new.express.adobe.com", "express.adobe.com",
    # IPFS gateways (decentralised hosting, heavily abused)
    "ipfs.io", "cloudflare-ipfs.com", "cf-ipfs.com", "ipfs.eth.aragon.network",
    "nftstorage.link", "dweb.link", "w3s.link",
    # Link shorteners (hide the real destination)
    "bit.ly", "tinyurl.com", "rebrand.ly", "surl.li", "t.co", "cutt.ly",
    "is.gd", "shorturl.at", "rb.gy", "ow.ly", "tiny.cc", "linktr.ee",
    # Messaging links
    "t.me", "wa.me",
}

# Domain endings often abused by phishers (some were free to register).
RISKY_TLDS = {"tk", "ml", "ga", "cf", "gq", "top", "xyz", "zip", "icu", "buzz",
              "rest", "cyou", "click", "link", "work", "support", "live"}

# Brand names scammers like to copy. If a domain CONTAINS one of these
# but is NOT the official site, that's a lookalike warning.
BRAND_KEYWORDS = {"gcash", "maya", "paymaya", "bdo", "bpi", "metrobank",
                  "landbank", "unionbank", "shopee", "lazada", "paypal",
                  "facebook", "instagram", "netflix", "apple", "microsoft",
                  "google", "amazon", "sss", "philhealth", "pagibig"}

SUSPICIOUS_WORDS = {"login", "signin", "verify", "secure", "account", "update",
                    "confirm", "wallet", "bank", "promo", "free", "claim",
                    "reward", "bonus", "unlock", "support", "recover"}


def extract_domain(url):
    """Same cleaning as Step 2: 'https://www.Site.com/login' -> 'site.com'."""
    url = str(url).strip()
    if "://" not in url:
        url = "http://" + url
    try:
        domain = urlparse(url).hostname or ""
    except ValueError:
        return ""
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _matches(domain, domain_set):
    """True if domain equals, or is a subdomain of, something in the set."""
    return any(domain == d or domain.endswith("." + d) for d in domain_set)


def is_trusted(domain):
    if _matches(domain, USER_HOSTING_PLATFORMS):
        return False
    return _matches(domain, TRUSTED_DOMAINS)


def _is_ip(domain):
    try:
        ipaddress.ip_address(domain)
        return True
    except ValueError:
        return False


def warning_signs(url, domain):
    """Rule-based warning signs a human can understand."""
    signs = []
    tld = domain.rsplit(".", 1)[-1] if "." in domain else ""
    parts = domain.split(".")

    if _is_ip(domain):
        signs.append("Uses a raw IP address instead of a website name.")
    if tld in RISKY_TLDS:
        signs.append(f"Ends in '.{tld}', a domain ending often abused by scammers.")
    if _matches(domain, USER_HOSTING_PLATFORMS):
        signs.append("Hosted on a free website builder where anyone can make a page.")
    if domain.startswith("xn--") or ".xn--" in domain:
        signs.append("Uses special characters that can imitate real letters (punycode).")
    if "@" in url:
        signs.append("Contains '@', which can hide the real destination.")
    if not str(url).strip().lower().startswith("https://") and "://" in url:
        signs.append("Does not use HTTPS (no secure connection).")

    # On hosting platforms, only check the part the page creator chose
    # ("gcash-claim.web.app" -> "gcash-claim"), not the platform's own name.
    owner_part = domain
    for platform in USER_HOSTING_PLATFORMS:
        if domain == platform or domain.endswith("." + platform):
            owner_part = domain[: -len(platform)].rstrip(".")
            break

    brands = sorted(b for b in BRAND_KEYWORDS
                    if b in owner_part.replace("-", "").replace(".", ""))
    if brands and not is_trusted(domain):
        signs.append(f"Mentions '{brands[0]}' but is NOT the official website.")

    words = sorted(w for w in SUSPICIOUS_WORDS if w in owner_part)
    if words:
        signs.append("Website name contains words common in scams: "
                     + ", ".join(words) + ".")

    # The part after the domain, e.g. "/view/gcash-claim"
    url_lower = str(url).lower()
    path = url_lower.split(domain, 1)[-1] if domain in url_lower else ""
    path_words = sorted(w for w in SUSPICIOUS_WORDS | BRAND_KEYWORDS
                        if w in path and len(w) > 3)
    if path_words and not is_trusted(domain):
        signs.append("The link's path contains suspicious words: "
                     + ", ".join(path_words) + ".")

    if domain.count("-") >= 3:
        signs.append(f"Has many hyphens ({domain.count('-')}).")
    if sum(c.isdigit() for c in domain) >= 5 and not _is_ip(domain):
        signs.append("Has many numbers in the website name.")
    if len(parts) >= 5:
        signs.append(f"Has many subdomain levels ({len(parts) - 2}).")
    if len(domain) > 40:
        signs.append(f"The website name is unusually long ({len(domain)} characters).")
    return signs


class PhishingDetector:
    def __init__(self, model_path=MODEL_PATH):
        self.model = joblib.load(model_path)
        self.tfidf = self.model.named_steps["tfidf"]
        self.vocab = np.array(self.tfidf.get_feature_names_out())
        # Average the weights of the 3 calibrated SVMs (same as Step 4)
        self.weights = np.mean(
            [c.estimator.coef_[0]
             for c in self.model.named_steps["clf"].calibrated_classifiers_], axis=0)

    def _top_pieces(self, domain, n=5):
        """Which character pieces of THIS domain pushed the AI each way."""
        vec = self.tfidf.transform([domain])
        idx = vec.indices
        contrib = vec.data * self.weights[idx]
        order = np.argsort(contrib)
        phishing = [(self.vocab[idx[i]], float(contrib[i]))
                    for i in order[::-1][:n] if contrib[i] > 0]
        legit = [(self.vocab[idx[i]], float(contrib[i]))
                 for i in order[:n] if contrib[i] < 0]
        return phishing, legit

    def analyze_url(self, url):
        domain = extract_domain(url)
        if not domain or ("." not in domain and not _is_ip(domain)):
            return {"valid": False, "url": url}

        result = {"valid": True, "url": url, "domain": domain,
                  "signs": warning_signs(url, domain)}

        adult_reason = adult_check(domain)

        # Known adult sites / adult domain endings: blocked right away.
        if adult_reason and "keywords" not in adult_reason:
            result["signs"].insert(0, adult_reason)
            result.update(verdict="Adult content", source="Adult content filter",
                          probability=0.0, phishing_pieces=[], legit_pieces=[])
            return result

        if domain in SHARED_HOSTS:
            result.update(verdict="Check carefully", source="Shared platform",
                          probability=0.0, phishing_pieces=[], legit_pieces=[])
            return result

        if is_trusted(domain):
            result.update(verdict="Legitimate", source="Trusted list",
                          probability=0.0, phishing_pieces=[], legit_pieces=[])
            return result

        prob = float(self.model.predict_proba([domain])[0, 1])
        if prob >= PHISHING_THRESHOLD:
            verdict = "Phishing"
        elif prob >= SUSPICIOUS_THRESHOLD:
            verdict = "Suspicious"
        else:
            verdict = "Likely legitimate"
        phishing_pieces, legit_pieces = self._top_pieces(domain)
        result.update(verdict=verdict, source="AI model", probability=prob,
                      phishing_pieces=phishing_pieces, legit_pieces=legit_pieces)

        # Adult KEYWORDS are checked after the AI: if the AI says phishing,
        # "Phishing" wins because it is the more dangerous result
        # (scammers sometimes use adult words as bait).
        if adult_reason:
            result["signs"].insert(0, adult_reason)
            if verdict != "Phishing":
                result.update(verdict="Adult content", source="Adult content filter")
        return result


if __name__ == "__main__":
    # Quick test:  python detector.py
    detector = PhishingDetector()
    tests = ["https://www.shopee.ph/cart", "m.facebook.com",
             "https://sites.google.com/view/gcash-claim",
             "http://secure-bdo-online.weebly.com/login",
             "paypal-verify-account.xyz", "https://www.uni-mainz.de",
             "http://192.168.1.5/login", "https://bit.ly/3xYz",
             "https://docs.google.com/forms/d/e/abc/viewform",
             "https://www.pornhub.com", "example.xxx", "hdpornvideo.co",
             "middlesex.ca", "jeuxvideo.com", "not a url"]
    for t in tests:
        r = detector.analyze_url(t)
        if not r["valid"]:
            print(f"{t:<45} -> invalid URL")
            continue
        print(f"{t:<45} -> {r['verdict']:<18} {r['probability']:6.1%}  ({r['source']})")
        for s in r["signs"]:
            print(f"{'':<48}! {s}")