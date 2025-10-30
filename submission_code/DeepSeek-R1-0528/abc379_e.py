import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    N = int(data[0].strip())
    S = data[1].strip()
    
    S2 = 0
    for i in range(N):
        S2 += (i + 1) * int(S[i])
    
    L = N + 20
    arr = [0] * L
    
    for k in range(1, N + 1):
        i = N - k
        d_val = int(S[i])
        A = (i + 1) * d_val
        arr[k] += A
        
    carry = 0
    for idx in range(L):
        total = arr[idx] + carry
        carry = total // 10
        arr[idx] = total % 10
        
    last_index = L - 1
    while last_index > 0 and arr[last_index] == 0:
        last_index -= 1
        
    if last_index < 0:
        s = "0"
    else:
        parts = []
        for i in range(last_index, -1, -1):
            parts.append(str(arr[i]))
        s = ''.join(parts)
    
    T_big = int(s) if s != "" else 0
    ans = (T_big - S2) // 9
    print(ans)

if __name__ == "__main__":
    main()