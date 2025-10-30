# YOUR CODE HERE
import sys

def longest_common_prefix(s1, s2):
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return i
    return min_len

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    strings = data[1:]
    
    total = 0
    for i in range(N-1):
        for j in range(i+1, N):
            total += longest_common_prefix(strings[i], strings[j])
    
    print(total)

if __name__ == "__main__":
    main()