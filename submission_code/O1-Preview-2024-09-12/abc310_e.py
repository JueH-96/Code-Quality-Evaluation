# YOUR CODE HERE

def main():
    import sys
    import threading
    def solve():
        import sys
        N = int(sys.stdin.readline())
        S = sys.stdin.readline().strip()
        A = [int(c) for c in S]
        total = 0
        N = len(A)
        i = 0
        while i < N:
            if A[i] == 0:
                # Handle run of zeros
                # For A_i=0, f(i,i)=0, f(i,j)=1 for j>i
                total += 0  # f(i,i) = 0
                zeros_len = 1
                i +=1
                total += (N - i +1)  # total ones from f(i,j) where j>i
                continue
            else:
                # Handle run of ones
                # For sequences of ones, f(i,i)=1, f(i,j) alternates between 1 and 0
                ones_len = 0
                j = i
                while j < N and A[j]==1:
                    ones_len +=1
                    j +=1
                # For each starting position in the run
                # The number of ones is floor(k+1)/2 for k from 1 to ones_len
                # Total ones in this run can be computed
                total_ones = 0
                for k in range(1, ones_len+1):
                    total_ones += (k+1)//2
                total += total_ones
                i = j
        print(total)
    threading.Thread(target=solve).start()