def solve():
    n, q = map(int, input().split())
    s = input()
    prefix_ones = [0] * (n + 1)
    prefix_slashes = [0] * (n + 1)
    prefix_twos = [0] * (n + 1)
    for i in range(n):
        prefix_ones[i+1] = prefix_ones[i]
        prefix_slashes[i+1] = prefix_slashes[i]
        prefix_twos[i+1] = prefix_twos[i]
        if s[i] == '1':
            prefix_ones[i+1] += 1
        elif s[i] == '/':
            prefix_slashes[i+1] += 1
        elif s[i] == '2':
            prefix_twos[i+1] += 1
            
    results = []
    for _ in range(q):
        l, r = map(int, input().split())
        max_len = 0
        for j in range(l, r + 1):
            if s[j-1] == '/':
                count1_prefix = max(0, prefix_ones[j-1-1] - prefix_ones[l-1]) if j > l else 0
                count2_suffix = max(0, prefix_twos[r] - prefix_twos[j]) if j < r else 0
                current_len = 2 * min(count1_prefix, count2_suffix) + 1
                max_len = max(max_len, current_len)
        results.append(max_len)
        
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()