def main():
    import sys

    # Read input string S (it is guaranteed there is one string in input)
    data = sys.stdin.read().split()
    if not data:
        return
    S = data[0]

    # We want to count the number of triples (i, j, k) with 1 <= i < j < k <= |S|
    # such that the 3-character string S[i]S[j]S[k] is a palindrome.
    # For a three-letter string to be a palindrome, the first and last characters must be the same.
    #
    # That is, we need S[i] == S[k], and j can be any index between i and k.
    #
    # Therefore, for each letter (from 'A' to 'Z'), let "pos" be the list of 1-indexed positions
    # where that letter appears. For any pair (i, k) from pos with i < k, there are (k - i - 1)
    # valid choices for j.
    #
    # A naive nested loop over all pairs will be too slow if S is all one letter (m could be 200,000).
    #
    # We can instead sum over each letter:
    #    Sum_{i < k in pos} (pos[k] - pos[i] - 1)
    #
    # Notice that:
    #    Sum_{i < k} (pos[k] - pos[i] - 1) 
    #      =  Sum_{i < k} (pos[k] - pos[i])
    #         - (number of pairs)
    #
    # The number of pairs is comb(m,2) = m*(m-1)//2.
    # 
    # We can compute Sum_{i < k} (pos[k] - pos[i]) in O(m) time using a summation trick:
    #    Sum_{i < k} (pos[k] - pos[i]) = sum_{k=0}^{m-1} pos[k]*(2*k - (m-1))
    # (Here we treat pos as 0-indexed but the positions stored are 1-indexed.)
    
    positions = {chr(c): [] for c in range(ord('A'), ord('Z')+1)}
    
    # Record positions (using 1-indexing as in the problem)
    for index, ch in enumerate(S, start=1):
        positions[ch].append(index)
        
    total = 0
    for ch, pos_list in positions.items():
        m = len(pos_list)
        if m < 2:
            continue
        # Compute sum_{i < k} (pos[k] - pos[i])
        current_sum = 0
        for i, pos in enumerate(pos_list):
            current_sum += pos * (2 * i - (m - 1))
        # Subtract the -1 for each pair: total number of pairs is m*(m-1)//2
        total += current_sum - (m * (m - 1) // 2)
        
    sys.stdout.write(str(total))

if __name__ == '__main__':
    main()