def longest_common_prefix_length(s1, s2):
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return i
    return min_len

def sum_of_longest_common_prefixes(strings):
    N = len(strings)
    total_sum = 0
    for i in range(N-1):
        for j in range(i+1, N):
            total_sum += longest_common_prefix_length(strings[i], strings[j])
    return total_sum

def main():
    N = int(input())
    strings = input().split()
    print(sum_of_longest_common_prefixes(strings))

if __name__ == "__main__":
    main()