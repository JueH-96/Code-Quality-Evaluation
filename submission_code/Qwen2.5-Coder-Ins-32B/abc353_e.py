# YOUR CODE HERE
import sys

def longest_common_prefix(s1, s2):
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return i
    return min_len

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    strings = input[1:]
    
    total_sum = 0
    for i in range(N):
        for j in range(i + 1, N):
            total_sum += longest_common_prefix(strings[i], strings[j])
    
    print(total_sum)

if __name__ == "__main__":
    main()