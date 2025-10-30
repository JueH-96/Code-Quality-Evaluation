import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        print(0)
        return
    
    first_line = data[0].split()
    if not first_line:
        print(0)
        return
        
    N = int(first_line[0])
    T_prime = first_line[1]
    m = len(T_prime)
    
    candidates = []
    for i in range(1, 1 + N):
        candidates.append(data[i].strip())
    
    valid_indices = []
    
    for idx, s in enumerate(candidates):
        n = len(s)
        if n == m:
            if s == T_prime:
                valid_indices.append(idx + 1)
            else:
                diff_count = 0
                for j in range(n):
                    if s[j] != T_prime[j]:
                        diff_count += 1
                        if diff_count > 1:
                            break
                if diff_count == 1:
                    valid_indices.append(idx + 1)
        elif n == m + 1:
            i_ptr, j_ptr = 0, 0
            diff_count = 0
            while j_ptr < m and i_ptr < n:
                if s[i_ptr] == T_prime[j_ptr]:
                    i_ptr += 1
                    j_ptr += 1
                else:
                    if diff_count == 0:
                        diff_count = 1
                        i_ptr += 1
                    else:
                        break
            if j_ptr == m:
                if diff_count == 1 or (diff_count == 0 and i_ptr == n - 1):
                    valid_indices.append(idx + 1)
        elif n == m - 1:
            i_ptr, j_ptr = 0, 0
            diff_count = 0
            while i_ptr < n and j_ptr < m:
                if s[i_ptr] == T_prime[j_ptr]:
                    i_ptr += 1
                    j_ptr += 1
                else:
                    if diff_count == 0:
                        diff_count = 1
                        j_ptr += 1
                    else:
                        break
            if i_ptr == n:
                if (diff_count == 1 and j_ptr == m) or (diff_count == 0 and j_ptr == m - 1):
                    valid_indices.append(idx + 1)
    
    k = len(valid_indices)
    print(k)
    if k > 0:
        print(" ".join(map(str, valid_indices)))

if __name__ == "__main__":
    main()