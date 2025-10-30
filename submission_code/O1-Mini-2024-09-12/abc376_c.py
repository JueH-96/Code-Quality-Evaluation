# YOUR CODE HERE
import sys

def main():
    import sys
    import threading

    def solve():
        import sys

        N_and_rest = sys.stdin.read().split()
        N = int(N_and_rest[0])
        A = list(map(int, N_and_rest[1:N+1]))
        B = list(map(int, N_and_rest[N+1:N+1+N-1]))
        
        A_sorted = sorted(A)
        B_sorted = sorted(B)
        
        prefix_ok = [False] * (N-1)
        for j in range(N-1):
            if j < len(B_sorted):
                prefix_ok[j] = B_sorted[j] >= A_sorted[j]
        
        prefix_ok_prefix = [False] * (N-1)
        for j in range(N-1):
            if j ==0:
                prefix_ok_prefix[j] = prefix_ok[j]
            else:
                prefix_ok_prefix[j] = prefix_ok_prefix[j-1] and prefix_ok[j]
        
        suffix_ok = [False] * (N-1)
        for j in range(N-1):
            suffix_ok[j] = B_sorted[j] >= A_sorted[j+1]
        
        suffix_ok_suffix = [False] * (N-1)
        for j in reversed(range(N-1)):
            if j == N-2:
                suffix_ok_suffix[j] = suffix_ok[j]
            else:
                suffix_ok_suffix[j] = suffix_ok[j] and suffix_ok_suffix[j+1]
        
        min_x = -1
        for k in range(N):
            condition1 = True
            if k >0:
                condition1 = prefix_ok_prefix[k-1]
            condition2 = True
            if k < N-1:
                condition2 = suffix_ok_suffix[k]
            if condition1 and condition2:
                if min_x == -1 or A_sorted[k] < min_x:
                    min_x = A_sorted[k]
        
        print(min_x)

    threading.Thread(target=solve).start()

if __name__ == "__main__":
    main()