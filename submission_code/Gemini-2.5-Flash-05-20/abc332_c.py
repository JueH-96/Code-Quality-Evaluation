import sys

def solve():
    # Read N and M from the first line
    N, M = map(int, sys.stdin.readline().split())
    # Read the schedule string S from the second line
    S = sys.stdin.readline().strip()

    # m_shirts_available: Number of plain T-shirts currently clean and ready to wear.
    # Initially, all M plain T-shirts are available.
    m_shirts_available = M

    # logo_shirts_available: Number of AtCoder logo T-shirts currently clean and ready to wear.
    # These shirts are from the total pool of logo shirts Takahashi "owns".
    # Initially, Takahashi owns 0 logo T-shirts, so 0 are available.
    logo_shirts_available = 0

    # max_logo_shirts_owned: This variable will store the maximum number of
    # AtCoder logo T-shirts Takahashi needed to own at any point.
    # This is our final answer. It starts at 0, as he initially owns none.
    max_logo_shirts_owned = 0

    # Simulate day by day
    for day_plan in S:
        if day_plan == '0':
            # Day with no plans: Wash all worn T-shirts.
            # All plain T-shirts become available again.
            m_shirts_available = M
            # All logo T-shirts Takahashi owns become available again.
            logo_shirts_available = max_logo_shirts_owned
        elif day_plan == '1':
            # Day for a meal: Needs a plain or logo T-shirt.
            # Prioritize using a plain T-shirt if possible.
            if m_shirts_available > 0:
                m_shirts_available -= 1
            else:
                # No plain T-shirts available. Must use a logo T-shirt.
                # Check if an existing logo T-shirt is available.
                if logo_shirts_available > 0:
                    logo_shirts_available -= 1
                else:
                    # No plain or logo T-shirts available. Must buy a new logo T-shirt.
                    # This increases the total number of logo shirts needed to own.
                    # The newly bought shirt is immediately worn for today, so it
                    # doesn't increase 'logo_shirts_available' for current day's use.
                    max_logo_shirts_owned += 1
        elif day_plan == '2':
            # Day for competitive programming: Needs a logo T-shirt.
            # Check if an existing logo T-shirt is available.
            if logo_shirts_available > 0:
                logo_shirts_available -= 1
            else:
                # No logo T-shirts available. Must buy a new one.
                # This increases the total number of logo shirts needed to own.
                # The newly bought shirt is immediately worn for today.
                max_logo_shirts_owned += 1
    
    # Print the minimum number of logo T-shirts needed to buy.
    sys.stdout.write(str(max_logo_shirts_owned) + '
')

solve()