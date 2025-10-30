import math

def main():
    N = int(input())
    A = list(map(int, input().split()))

    # dp_even stores the maximum experience with an even number of defeated monsters.
    # dp_odd stores the maximum experience with an odd number of defeated monsters.

    # Base case: Before any monsters are encountered.
    # 0 monsters defeated (even count), score is 0.
    dp_even = 0
    # Impossible to have an odd number of defeated monsters. Use -infinity.
    dp_odd = -math.inf 

    for i in range(N):
        strength = A[i]

        # Store values from the previous iteration to use for current calculations
        prev_dp_even = dp_even
        prev_dp_odd = dp_odd

        # Calculate new dp_even for the current monster
        # Option 1: Don't defeat the current monster.
        #   Score remains prev_dp_even (count of defeated monsters is still even).
        current_even_opt1 = prev_dp_even
        
        # Option 2: Defeat the current monster. It becomes an even-th defeated monster.
        #   This implies an odd number of monsters were defeated before this one.
        #   Score is prev_dp_odd + 2 * strength.
        #   This path is only valid if prev_dp_odd is not -infinity.
        current_even_opt2 = -math.inf
        if prev_dp_odd != -math.inf:
            current_even_opt2 = prev_dp_odd + 2 * strength
        
        dp_even = max(current_even_opt1, current_even_opt2)

        # Calculate new dp_odd for the current monster
        # Option 1: Don't defeat the current monster.
        #   Score remains prev_dp_odd (count of defeated monsters is still odd).
        current_odd_opt1 = prev_dp_odd

        # Option 2: Defeat the current monster. It becomes an odd-th defeated monster.
        #   This implies an even number of monsters were defeated before this one.
        #   Score is prev_dp_even + strength.
        #   prev_dp_even is always >= 0 (starts at 0, can only increase or stay via max).
        current_odd_opt2 = prev_dp_even + strength
            
        dp_odd = max(current_odd_opt1, current_odd_opt2)
        
    # The final maximum experience is the best score ending with either an
    # even or odd number of defeated monsters.
    print(max(dp_even, dp_odd))

if __name__ == '__main__':
    main()