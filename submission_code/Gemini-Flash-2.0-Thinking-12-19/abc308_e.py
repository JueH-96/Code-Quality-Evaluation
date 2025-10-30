def calculate_mex(a, b, c):
    seen = set()
    seen.add(a)
    seen.add(b)
    seen.add(c)
    mex_value = 0
    while True:
        if mex_value not in seen:
            return mex_value
        mex_value += 1

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    s = input()
    
    m_indices = []
    e_indices = []
    x_indices = []
    
    for i in range(n):
        if s[i] == 'M':
            m_indices.append(i)
        elif s[i] == 'E':
            e_indices.append(i)
        elif s[i] == 'X':
            x_indices.append(i)
            
    total_mex_sum = 0
    
    for i_index in m_indices:
        for j_index in e_indices:
            for k_index in x_indices:
                if i_index < j_index < k_index:
                    current_mex = calculate_mex(a[i_index], a[j_index], a[k_index])
                    total_mex_sum += current_mex
                    
    print(total_mex_sum)

if __name__ == '__main__':
    solve()