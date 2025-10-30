import sys
from collections import defaultdict

def count_valid_pairs(n, a):
    # Count the frequency of each element in the array
    freq = defaultdict(int)
    for num in a:
        freq[num] += 1
    
    # Initialize the count of valid pairs
    valid_pairs = 0
    
    # First, handle the case where a_i == a_j
    # For each unique element, calculate the number of pairs where a_i == a_j
    for num in freq:
        count = freq[num]
        if count >= 2:
            valid_pairs += count * (count - 1) // 2
    
    # Second, handle the case where a_i != a_j but 2^a_i ^ 2^a_j == 2^a_j ^ 2^a_i
    # This happens when a_i * a_j == a_j * a_i, which is always true, but we need to find when 2^(a_i * 2^a_j) == 2^(a_j * 2^a_i)
    # Simplifying, we get a_i * 2^a_j == a_j * 2^a_i
    # Taking log2 on both sides: log2(a_i) + a_j == log2(a_j) + a_i
    # Rearranging: log2(a_i) - a_i == log2(a_j) - a_j
    # Since a_i and a_j are integers, log2(a_i) is log2(a_i) only if a_i is a power of 2
    # So, we need to find pairs where a_i and a_j are both powers of 2 and a_i == a_j
    # But since we already handled the case where a_i == a_j, we need to find pairs where a_i and a_j are different powers of 2
    # However, for a_i and a_j being different powers of 2, 2^a_i ^ 2^a_j != 2^a_j ^ 2^a_i
    # So, no additional pairs are valid in this case
    
    return valid_pairs

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    t = int(data[idx])
    idx += 1
    for _ in range(t):
        n = int(data[idx])
        idx += 1
        a = list(map(int, data[idx:idx+n]))
        idx += n
        print(count_valid_pairs(n, a))

if __name__ == "__main__":
    main()