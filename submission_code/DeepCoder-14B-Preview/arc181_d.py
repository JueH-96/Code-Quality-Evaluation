import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    A = list(map(int, data[3:3+N]))
    
    sum_A = sum(A)
    remaining_total = K - sum_A
    if remaining_total < 0:
        remaining_total = 0
    
    result = []
    
    for i in range(N):
        j_list = [A[j] for j in range(N) if j != i]
        j_list.sort()
        prefix = [0]
        for x in j_list:
            prefix.append(prefix[-1] + x)
        
        X_max = remaining_total
        low = 0
        high = X_max
        ans = -1
        
        while low <= high:
            mid = (low + high) // 2
            B = A[i] + mid
            R = remaining_total - mid
            if R < 0:
                high = mid - 1
                continue
            
            # Calculate count_high
            count_high = len(j_list) - bisect.bisect_right(j_list, B)
            
            # Calculate count_low and sum_Aj_low
            count_low = bisect.bisect_left(j_list, B)
            sum_Aj_low = prefix[count_low]
            sum_x_j_min_low = (B + 1) * count_low - sum_Aj_low
            
            # Calculate count_eq
            left = bisect.bisect_left(j_list, B)
            right = bisect.bisect_right(j_list, B)
            count_eq = right - left
            
            sum_x_j_min = sum_x_j_min_low + count_eq
            
            if sum_x_j_min <= R:
                possible_j = count_low + count_eq
            else:
                x_j_mins = [B - x + 1 for x in j_list[:count_low]]
                x_j_mins.sort()
                sum_so_far = 0
                possible_low = 0
                for x in x_j_mins:
                    if sum_so_far + x <= R:
                        sum_so_far += x
                        possible_low += 1
                    else:
                        break
                remaining = R - sum_so_far
                possible_eq = min(remaining, count_eq)
                possible_j = possible_low + possible_eq
            
            total_j_above = count_high + possible_j
            if total_j_above < M:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        
        # Check if X=0 is acceptable if ans was not found
        if ans == -1:
            B = A[i]
            R = remaining_total
            if R >= 0:
                count_high = len(j_list) - bisect.bisect_right(j_list, B)
                count_low = bisect.bisect_left(j_list, B)
                sum_Aj_low = prefix[count_low]
                sum_x_j_min_low = (B + 1) * count_low - sum_Aj_low
                left = bisect.bisect_left(j_list, B)
                right = bisect.bisect_right(j_list, B)
                count_eq = right - left
                
                sum_x_j_min = sum_x_j_min_low + count_eq
                if sum_x_j_min <= R:
                    possible_j = count_low + count_eq
                else:
                    x_j_mins = [B - x + 1 for x in j_list[:count_low]]
                    x_j_mins.sort()
                    sum_so_far = 0
                    possible_low = 0
                    for x in x_j_mins:
                        if sum_so_far + x <= R:
                            sum_so_far += x
                            possible_low += 1
                        else:
                            break
                    remaining = R - sum_so_far
                    possible_eq = min(remaining, count_eq)
                    possible_j = possible_low + possible_eq
                
                total_j_above = count_high + possible_j
                if total_j_above < M:
                    ans = 0
        
        # Determine the final result for this candidate
        if ans != -1 and ans <= remaining_total:
            result.append(str(ans))
        else:
            result.append('-1' if remaining_total == 0 or ans == -1 else str(ans))
    
    print(' '.join(result))

if __name__ == '__main__':
    main()