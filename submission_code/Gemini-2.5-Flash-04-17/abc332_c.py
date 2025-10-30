# YOUR CODE HERE
import sys

# Function to check if B logo shirts are sufficient
def check(N, M, S, B):
    # current_plain: available plain shirts in the current wash cycle
    # current_logo: available logo shirts in the current wash cycle
    # Initially, all M plain shirts and B logo shirts are available.
    current_plain = M
    current_logo = B

    for i in range(N):
        event = S[i]

        if event == '0':
            # Wash day: All shirts become available again.
            # The number of available shirts resets to the total owned counts.
            # Plain shirts reset to M, logo shirts reset to B.
            current_plain = M
            current_logo = B
        elif event == '1':
            # Meal day: Need plain or logo. Prioritize plain.
            if current_plain > 0:
                current_plain -= 1
            elif current_logo > 0:
                current_logo -= 1
            else:
                # Need shirt but none available (neither plain nor logo)
                return False
        elif event == '2':
            # Event day: Need logo only.
            if current_logo > 0:
                current_logo -= 1
            else:
                # Need logo shirt but none available
                return False

    # Successfully completed all days
    return True

def solve():
    # Read N and M from the first line
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    M = int(line1[1])

    # Read the schedule string S
    S = sys.stdin.readline().strip()

    # Binary search for the minimum number of logo shirts to buy (B)
    # The minimum possible B is 0.
    # The maximum possible B is N (a safe upper bound, as you need at most 1 shirt per day,
    # and in the worst case, all days require a logo shirt and no washes occur).
    low = 0
    high = N
    ans = N # Initialize with a value that is sufficient (or higher)

    while low <= high:
        mid = (low + high) // 2 # Number of logo shirts to test

        if check(N, M, S, mid):
            # mid logo shirts are sufficient, try fewer
            ans = mid
            high = mid - 1
        else:
            # mid logo shirts are not sufficient, need more
            low = mid + 1

    # The minimum number of logo shirts needed is stored in ans
    print(ans)

solve()