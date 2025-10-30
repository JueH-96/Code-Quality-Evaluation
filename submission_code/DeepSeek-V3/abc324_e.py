# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    T = data[1]
    S_list = data[2:2+N]
    
    # Precompute the prefix and suffix counts for each S
    prefix_counts = []
    suffix_counts = []
    for S in S_list:
        # Compute prefix counts: how many characters of T can be matched in S
        prefix = [0] * (len(T) + 1)
        ptr = 0
        for c in S:
            if ptr < len(T) and c == T[ptr]:
                ptr += 1
            prefix[ptr] += 1
        prefix_counts.append(prefix)
        
        # Compute suffix counts: how many characters of T can be matched in reverse in S
        suffix = [0] * (len(T) + 1)
        ptr = len(T) - 1
        for c in reversed(S):
            if ptr >= 0 and c == T[ptr]:
                ptr -= 1
            suffix[len(T) - 1 - ptr] += 1
        suffix_counts.append(suffix)
    
    # Now, for each pair (i,j), we need to find the number of pairs where prefix_counts[i][k] > 0 and suffix_counts[j][len(T) - k] > 0
    # To optimize, we can precompute the counts for each k
    # Create a dictionary to count the number of S_i that have prefix_counts[i][k] > 0 for each k
    prefix_k_counts = {}
    for i in range(N):
        for k in range(len(T) + 1):
            if prefix_counts[i][k] > 0:
                if k not in prefix_k_counts:
                    prefix_k_counts[k] = 0
                prefix_k_counts[k] += 1
    
    # Similarly, create a dictionary to count the number of S_j that have suffix_counts[j][len(T) - k] > 0 for each k
    suffix_k_counts = {}
    for j in range(N):
        for k in range(len(T) + 1):
            if suffix_counts[j][k] > 0:
                if k not in suffix_k_counts:
                    suffix_k_counts[k] = 0
                suffix_k_counts[k] += 1
    
    # Now, for each k, the number of pairs (i,j) where prefix_counts[i][k] > 0 and suffix_counts[j][len(T) - k] > 0 is prefix_k_counts[k] * suffix_k_counts[len(T) - k]
    total = 0
    for k in range(len(T) + 1):
        if k in prefix_k_counts and (len(T) - k) in suffix_k_counts:
            total += prefix_k_counts[k] * suffix_k_counts[len(T) - k]
    
    print(total)

if __name__ == "__main__":
    main()