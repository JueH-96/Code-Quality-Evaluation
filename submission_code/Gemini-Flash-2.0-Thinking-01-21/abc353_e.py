def solve():
    n = int(input())
    strings = input().split()
    
    def get_lcp_length(s1, s2):
        length = 0
        for i in range(min(len(s1), len(s2))):
            if s1[i] == s2[i]:
                length += 1
            else:
                break
        return length
        
    total_lcp_sum = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_lcp_sum += get_lcp_length(strings[i], strings[j])
            
    print(total_lcp_sum)

if __name__ == '__main__':
    solve()