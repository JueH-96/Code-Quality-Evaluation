import sys
import bisect

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    q = int(data[1])
    a_list = list(map(int, data[2:2+n]))
    a_list.sort()
    minA = a_list[0]
    maxA = a_list[-1]
    
    index = 2 + n
    out_lines = []
    for _ in range(q):
        b = int(data[index])
        k = int(data[index+1])
        index += 2
        lo = 0
        hi = max(abs(b - minA), abs(b - maxA))
        
        while lo < hi:
            mid = (lo + hi) // 2
            low_bound = b - mid
            high_bound = b + mid
            left_index = bisect.bisect_left(a_list, low_bound)
            right_index = bisect.bisect_right(a_list, high_bound)
            count = right_index - left_index
            
            if count >= k:
                hi = mid
            else:
                lo = mid + 1
                
        out_lines.append(str(lo))
    
    print("
".join(out_lines))

if __name__ == "__main__":
    main()