import sys

def main():
    import threading
    def run():
        input = sys.stdin.read
        data = input().split()
        idx = 0
        T = int(data[idx])
        idx += 1
        results = []
        for _ in range(T):
            N = int(data[idx])
            idx += 1
            A = []
            B = []
            C = []
            x_max_list = []
            for _ in range(N):
                a = int(data[idx])
                b = int(data[idx+1])
                c = int(data[idx+2])
                idx += 3
                A.append(a)
                B.append(b)
                C.append(c)
                # Compute x_max_i = (C_i - B_i -1) // A_i
                numerator = c - b - 1
                if numerator < 1:
                    x_max_i = 0
                else:
                    x_max_i = numerator // a
                x_max_list.append(x_max_i)
            x_max = min(x_max_list)
            if x_max < 1:
                results.append('0')
                continue
            ans = 0
            # Iterate over x in 1 to x_max
            # To optimize, but in worst case, this is O(N * x_max)
            for x in range(1, x_max + 1):
                min_y = None
                valid = True
                for i in range(N):
                    a = A[i]
                    b = B[i]
                    c = C[i]
                    rhs = c - a * x
                    if rhs <= b:
                        valid = False
                        break
                    max_y_i = (rhs - 1) // b
                    if max_y_i < 1:
                        valid = False
                        break
                    if min_y is None or max_y_i < min_y:
                        min_y = max_y_i
                if valid and min_y is not None:
                    ans += min_y
            results.append(str(ans))
        print('
'.join(results))
    threading.Thread(target=run).start()

if __name__ == "__main__":
    main()