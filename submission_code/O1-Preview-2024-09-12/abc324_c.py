# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_T_prime = sys.stdin.readline().strip().split()
    while len(N_and_T_prime) <2:
        line = sys.stdin.readline().strip()
        if line == '':
            break
        N_and_T_prime += line.split()
    N = int(N_and_T_prime[0])
    T_prime = N_and_T_prime[1]
    N_list = []
    total_S_length = 0
    for _ in range(N):
        s = sys.stdin.readline().strip()
        while s == '':
            s = sys.stdin.readline().strip()
        N_list.append(s)
        total_S_length += len(s)
        if total_S_length > 5 * 10**5:
            break

    T_prime_len = len(T_prime)

    result_indices = []
    T_p = T_prime
    T_p_len = T_prime_len
    for idx, s in enumerate(N_list):
        s_len = len(s)
        diff_len = T_p_len - s_len
        possible = False
        if diff_len == 0:
            mismatches = 0
            for c1, c2 in zip(s, T_p):
                if c1 != c2:
                    mismatches +=1
                    if mismatches >1:
                        break
            if mismatches == 0 or mismatches ==1:
                possible = True
        elif diff_len == 1:
            # Check if inserting one character into s gives T_p
            i = j = 0
            mismatch_occurred = False
            while i < s_len and j < T_p_len:
                if s[i] == T_p[j]:
                    i +=1
                    j +=1
                else:
                    if mismatch_occurred:
                        break
                    else:
                        mismatch_occurred = True
                        j +=1  # Skip one character in T_p
            else:
                # After loop
                if mismatch_occurred or (j == T_p_len -1 and i == s_len):
                    possible = True
        elif diff_len == -1:
            # Check if deleting one character from s gives T_p
            i = j = 0
            mismatch_occurred = False
            while i < s_len and j < T_p_len:
                if s[i] == T_p[j]:
                    i +=1
                    j +=1
                else:
                    if mismatch_occurred:
                        break
                    else:
                        mismatch_occurred = True
                        i +=1  # Skip one character in s
            else:
                # After loop
                if mismatch_occurred or (i == s_len -1 and j == T_p_len):
                    possible = True
        if possible:
            result_indices.append(idx+1)  # indices are 1-based

    print(len(result_indices))
    if result_indices:
        print(' '.join(map(str, result_indices)))

threading.Thread(target=main).start()