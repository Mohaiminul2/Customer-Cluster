# Shared constants for the Customer Segmentation project
# ---------------------------------------------------------------------------
# Colour palette (dark theme)
NAVY = "#0F1E35"
NAVY_CARD = "#162032"
NAVY_BORDER = "#1E2D45"
WHITE = "#F0F4F8"
MUTED = "#6B7E96"
GREEN = "#22C55E"
TEAL = "#0D9488"
BLUE = "#3B82F6"
AMBER = "#F59E0B"
RED = "#EF4444"
GREY = "#64748B"

# Segment colour mapping (business labels)
SEGMENT_COLORS = {
    "Champions": GREEN,
    "Loyal Customers": TEAL,
    "Potential Loyalists": BLUE,
    "At Risk": AMBER,
    "Lost / Churned": RED,
}

# Segment order – used for consistent ordering in tables & charts
SEGMENT_ORDER = ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Lost / Churned"]

# Optional icons for UI display
SEGMENT_ICONS = {
    "Champions": "🏆",
    "Loyal Customers": "💙",
    "Potential Loyalists": "🔵",
    "At Risk": "⚠️",
    "Lost / Churned": "🔴",
}
