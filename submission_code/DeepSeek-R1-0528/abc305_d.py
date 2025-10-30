import bisect

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    q = int(data[1+n])
    queries = []
    index = 1+n+1
    for i in range(q):
        l = int(data[index])
        r = int(data[index+1])
        index += 2
        queries.append((l, r))
    
    M = (n - 1) // 2
    start_times = []
    end_times = []
    for j in range(M):
        start_times.append(A[2*j+1])
        end_times.append(A[2*j+2])
    
    prefix_sleep = [0] * (M+1)
    for j in range(M):
        dur = end_times[j] - start_times[j]
        prefix_sleep[j+1] = prefix_sleep[j] + dur

    results = []
    for (l, r) in queries:
        j_left = bisect.bisect_left(end_times, l)
        j_right = bisect.bisect_right(start_times, r) - 1
        
        if j_left <= j_right:
            total = 0
            if j_left + 1 <= j_right:
                total += prefix_sleep[j_right] - prefix_sleep[j_left+1]
                
            s1 = start_times[j_left]
            e1 = end_times[j_left]
            overlap1 = min(e1, r) - max(s1, l)
            total += overlap1
            
            if j_left != j_right:
                s2 = start_times[j_right]
                e2 = end_times[j_right]
                overlap2 = min(e2, r) - max(s2, l)
                total += overlap2
                
            results.append(str(total))
        else:
            results.append("0")
            
    print("
".join(results))

if __name__ == "__main__":
    main()