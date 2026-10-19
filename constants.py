# Shared constants for the Customer Segmentation project
# ---------------------------------------------------------------------------
# Colour palette (dark theme)
NAVY = "#0B1220"          # page background
NAVY_CARD = "#111B2E"     # cards / sidebar
NAVY_RAISED = "#16233A"   # hover, inputs, tooltips
NAVY_BORDER = "#1F2D45"   # hairlines
WHITE = "#E8EEF7"         # primary text
MUTED = "#8DA2BE"         # secondary text
SUBTLE = "#5B6F8C"        # tertiary text, gridlines
ACCENT = "#6EA8FE"        # interactive accent

GREEN = "#34D399"
TEAL = "#2DD4BF"
BLUE = "#38BDF8"
VIOLET = "#A78BFA"
AMBER = "#FBBF24"
RED = "#F87171"
GREY = "#64748B"

# Diverging scale for "better / worse than typical" (heatmap + its key)
GOOD_TINT = "#2FA37A"
BAD_TINT = "#B4545E"
MID_TINT = "#1B2A42"

# Segment colour mapping (business labels). Ordered best -> worst.
SEGMENT_COLORS = {
    "Champions": GREEN,
    "Loyal Customers": BLUE,
    "Potential Loyalists": VIOLET,
    "At Risk": AMBER,
    "Lost / Churned": RED,
}

# Segment order – used for consistent ordering in tables & charts
SEGMENT_ORDER = ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Lost / Churned"]

# Optional icons for UI display
SEGMENT_ICONS = {
    "Champions": "🏆",
    "Loyal Customers": "💙",
    "Potential Loyalists": "🌱",
    "At Risk": "⚠️",
    "Lost / Churned": "💤",
}

# One-line plain-language definition per segment. Segments are value tiers:
# clusters are ranked best-to-worst on standardised Recency, Frequency and
# Monetary, then named in that order.
SEGMENT_DESCRIPTIONS = {
    "Champions": "Highest-value tier: very high spend and order count, bought recently.",
    "Loyal Customers": "Second tier: frequent, recent buyers with strong lifetime value.",
    "Potential Loyalists": "Third tier: recent buyers with solid order counts and room to grow.",
    "At Risk": "Mid-to-low value: ordered a few times, now drifting quiet.",
    "Lost / Churned": "Lowest tier: long inactive, mostly one or two orders.",
}

# Recommended playbook per segment. ``{currency}`` is filled at render time.
RECOMMENDATIONS = {
    "Champions": {
        "priority": "PROTECT",
        "color": GREEN,
        "actions": [
            "Launch a VIP loyalty tier with exclusive early access and perks",
            "Personalised thank-you communications and milestone rewards",
            "Referral programme - leverage their advocacy to acquire similar customers",
            "Monitor closely for any recency slip; trigger re-engagement immediately",
        ],
        "metric": "Champion retention rate >= 95% month-over-month",
    },
    "Loyal Customers": {
        "priority": "GROW",
        "color": BLUE,
        "actions": [
            "Frequency upsell - bundle deals and auto-replenishment nudges",
            "Targeted cross-sell based on purchase category history",
            "Bi-weekly personalised email cadence with curated recommendations",
            "Identify top 20% most likely to become Champions; give them targeted push",
        ],
        "metric": "Move 15-20% of Loyal customers into Champions tier per quarter",
    },
    "Potential Loyalists": {
        "priority": "CONVERT",
        "color": VIOLET,
        "actions": [
            "Lift order frequency toward the Loyal tier with bundles and replenishment reminders",
            "Cross-sell adjacent categories to raise basket size",
            "Keep them warm: trigger a nudge if recency passes 30 days",
            "Post-purchase review request to deepen engagement",
        ],
        "metric": "Move 15% of Potential Loyalists into the Loyal tier per quarter",
    },
    "At Risk": {
        "priority": "WIN BACK",
        "color": AMBER,
        "actions": [
            "3-step win-back email sequence over 21 days (soft -> offer -> final)",
            '"We miss you" discount offer (10-15%) to trigger re-engagement',
            "Short NPS survey to understand reason for disengagement",
            "Prioritise customers with 3+ orders for highest ROI on re-engagement",
        ],
        "metric": "Win-back rate >= 30% of targeted At Risk customers per campaign",
    },
    "Lost / Churned": {
        "priority": "ASSESS ROI",
        "color": RED,
        "actions": [
            "One-time deep-discount reactivation campaign (20-25%)",
            "Target only top 25% by historical monetary value - rest are not cost-effective",
            "Sunset low-value customers from active CRM lists to reduce cost",
            "Use profile data to improve retention signals for other segments",
        ],
        "metric": "Reactivation rate >= 15% - only for customers with LTV > {currency}200",
    },
}
