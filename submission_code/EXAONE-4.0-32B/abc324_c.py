import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        print(0)
        return
    
    first_line = data[0].split()
    N = int(first_line[0])
    T_prime = first_line[1].strip()
    L = len(T_prime)
    
    strings = []
    for i in range(1, 1 + N):
        s = data[i].strip()
        strings.append(s)
    
    valid_indices = []
    
    for idx, s in enumerate(strings, start=1):
        n = len(s)
        if n == L:
            diff_count = 0
            for i in range(n):
                if s[i] != T_prime[i]:
                    diff_count += 1
                    if diff_count > 1:
                        break
            if diff_count <= 1:
                valid_indices.append(idx)
                
        elif n == L - 1:
            i_ptr = 0
            j_ptr = 0
            skip_used = False
            while i_ptr < n and j_ptr < L:
                if s[i_ptr] == T_prime[j_ptr]:
                    i_ptr += 1
                    j_ptr += 1
                else:
                    if not skip_used:
                        skip_used = True
                        j_ptr += 1
                    else:
                        break
            if i_ptr == n:
                valid_indices.append(idx)
                
        elif n == L + 1:
            i_ptr = 0
            j_ptr = 0
            skip_used = False
            while i_ptr < n and j_ptr < L:
                if s[i_ptr] == T_prime[j_ptr]:
                    i_ptr += 1
                    j_ptr += 1
                else:
                    if not skip_used:
                        skip_used = True
                        i_ptr += 1
                    else:
                        break
            if j_ptr == L:
                valid_indices.append(idx)
                
    print(len(valid_indices))
    if valid_indices:
        print(" ".join(map(str, valid_indices)))

if __name__ == "__main__":
    main()