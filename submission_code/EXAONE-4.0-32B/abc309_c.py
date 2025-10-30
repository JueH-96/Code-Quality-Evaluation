import sys
import bisect

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    K = int(data[1])
    meds = []
    a_list = []
    index = 2
    for i in range(n):
        a = int(data[index])
        b = int(data[index+1])
        index += 2
        meds.append((a, b))
        a_list.append(a)
    
    meds.sort(key=lambda x: x[0])
    a_list.sort()
    
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + meds[i][1]
    
    def total_pills(day):
        pos = bisect.bisect_left(a_list, day)
        return suffix_sum[pos]
    
    low = 1
    high = a_list[-1] + 2 if n > 0 else 2
    
    while low < high:
        mid = (low + high) // 2
        pills_mid = total_pills(mid)
        if pills_mid <= K:
            high = mid
        else:
            low = mid + 1
            
    print(low)

if __name__ == "__main__":
    main()