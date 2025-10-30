def longest_common_prefix(s1, s2):
    """Returns the length of the longest common prefix of s1 and s2."""
    min_length = min(len(s1), len(s2))
    for i in range(min_length):
        if s1[i] != s2[i]:
            return i
    return min_length

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    strings = data[1:N+1]
    
    # Sort the strings
    strings.sort()
    
    # Calculate the sum of longest common prefixes of consecutive pairs
    total_lcp_sum = 0
    for i in range(N - 1):
        total_lcp_sum += longest_common_prefix(strings[i], strings[i + 1])
    
    print(total_lcp_sum)