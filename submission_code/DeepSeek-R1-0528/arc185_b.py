import sys

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        n = int(data[index]); index += 1
        A = list(map(int, data[index:index+n]))
        index += n
        
        if n == 2:
            if A[0] <= A[1]:
                results.append("Yes")
            else:
                results.append("No")
        else:
            total = sum(A)
            prefix = [0] * (n-1)
            prefix[0] = A[0]
            for i in range(1, n-1):
                prefix[i] = prefix[i-1] + A[i]
            
            low = A[0]
            high = total
            found = False
            while low <= high:
                mid = (low + high) // 2
                s = mid
                F_prev = mid
                valid = True
                for i in range(1, n-1):
                    required = prefix[i] - s
                    if required > F_prev:
                        F_i = required
                    else:
                        F_i = F_prev
                    if s + F_i > total:
                        valid = False
                        break
                    s += F_i
                    F_prev = F_i
                
                if not valid:
                    high = mid - 1
                else:
                    last = total - s
                    if last >= F_prev:
                        found = True
                        break
                    else:
                        low = mid + 1
            if found:
                results.append("Yes")
            else:
                results.append("No")
    
    for res in results:
        print(res)

if __name__ == "__main__":
    main()