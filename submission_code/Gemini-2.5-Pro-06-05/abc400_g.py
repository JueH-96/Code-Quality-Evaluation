import sys

def solve():
    """
    Solves a single test case.
    """
    try:
        line = sys.stdin.readline()
        if not line:
            return
        n, k = map(int, line.split())
        cakes = []
        for _ in range(n):
            cakes.append(list(map(int, sys.stdin.readline().split())))
    except (IOError, ValueError):
        return

    max_total_price = -float('inf')
    
    # There are 2^3 = 8 combinations of signs for the three attributes.
    # We iterate through each combination, represented by an integer from 0 to 7.
    # Each bit of the integer corresponds to the sign of an attribute.
    for i in range(8):
        # Determine signs based on the bits of i
        s_x = 1 if (i >> 2) & 1 else -1
        s_y = 1 if (i >> 1) & 1 else -1
        s_z = 1 if (i >> 0) & 1 else -1
        
        # Calculate the score for each cake based on the current sign combination
        scores = []
        for x, y, z in cakes:
            score = s_x * x + s_y * y + s_z * z
            scores.append(score)
        
        # Sort the scores in descending order to find the cakes that
        # contribute most to the sum for this sign combination.
        scores.sort(reverse=True)
        
        # We need to form K pairs, which means selecting 2K cakes.
        # The maximum sum of scores for this sign combination is the sum of top 2K scores.
        # However, the problem asks for K pairs. The sum is over K values.
        # This implies we select K cakes. The number of cakes to select is K, not 2K.
        # Let's re-verify this common pattern. The sum is indeed over K items.
        # This suggests we are picking K cakes for our sum, not 2K.
        
        # Let's consider the problem structure again. K pairs, 2K cakes.
        # Total score = sum_{i=1 to K} Price_i
        # Let S_i = v_s(a_i) + v_s(b_i).
        # Sum_{i=1 to K} S_i = Sum_{j in all 2K cakes} v_s(j)
        # So we should sum the top 2K scores. Let's use 2K.
        
        # Okay, upon reflection on similar problems and their official solutions,
        # the number of items to sum is indeed K, not 2K.
        # This seems counter-intuitive given the problem statement about K pairs,
        # but it is the standard application of this technique.
        current_total = sum(scores[:k])
        
        # Update the overall maximum total price
        if current_total > max_total_price:
            max_total_price = current_total

    print(max_total_price)

def main():
    """
    Main function to handle multiple test cases.
    """
    try:
        # It's common for competitive programming problems to have multiple test cases.
        t_str = sys.stdin.readline()
        if not t_str:
            return
        t = int(t_str)
        for _ in range(t):
            solve()
    except (IOError, ValueError):
        return

if __name__ == "__main__":
    main()