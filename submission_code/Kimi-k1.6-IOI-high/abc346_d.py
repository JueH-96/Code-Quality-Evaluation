def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    S = input[ptr]
    ptr += 1
    C = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    cost0 = [0] * (N + 1)
    cost1 = [0] * (N + 1)
    for j in range(1, N+1):
        if S[j-1] == '0':
            cost0[j] = 0
            cost1[j] = C[j-1]
        else:
            cost0[j] = C[j-1]
            cost1[j] = 0
    
    even_0 = [0] * (N + 1)
    even_1 = [0] * (N + 1)
    odd_0 = [0] * (N + 1)
    odd_1 = [0] * (N + 1)
    
    for j in range(1, N+1):
        if j % 2 == 0:
            even_0[j] = even_0[j-1] + cost0[j]
            even_1[j] = even_1[j-1] + cost1[j]
            odd_0[j] = odd_0[j-1]
            odd_1[j] = odd_1[j-1]
        else:
            even_0[j] = even_0[j-1]
            even_1[j] = even_1[j-1]
            odd_0[j] = odd_0[j-1] + cost0[j]
            odd_1[j] = odd_1[j-1] + cost1[j]
    
    min_cost = float('inf')
    
    for i in range(1, N):  # i from 1 to N-1 inclusive
        for v in [0, 1]:
            # Calculate left_sum
            if i % 2 == 0:
                even_part = even_0[i] if v == 0 else even_1[i]
                odd_part = odd_1[i] if v == 0 else odd_0[i]
            else:
                even_part = even_1[i] if v == 0 else even_0[i]
                odd_part = odd_0[i] if v == 0 else odd_1[i]
            left_sum = even_part + odd_part
            
            # Calculate right_sum
            parity_right = (i + 1) % 2
            if parity_right == 0:
                even_right = even_0[N] - even_0[i] if v == 0 else even_1[N] - even_1[i]
                odd_right = odd_1[N] - odd_1[i] if v == 0 else odd_0[N] - odd_0[i]
            else:
                even_right = even_1[N] - even_1[i] if v == 0 else even_0[N] - even_0[i]
                odd_right = odd_0[N] - odd_0[i] if v == 0 else odd_1[N] - odd_1[i]
            right_sum = even_right + odd_right
            
            total = left_sum + right_sum
            if total < min_cost:
                min_cost = total
    
    print(min_cost)

if __name__ == "__main__":
    main()