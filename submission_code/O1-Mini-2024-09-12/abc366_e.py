# YOUR CODE HERE
import sys, bisect

def get_sum_list(sorted_coord, D):
    N = len(sorted_coord)
    median = sorted_coord[N//2]
    sum_x = sum(abs(median - x) for x in sorted_coord)
    list_sum = [sum_x]
    
    # Iterate to the right
    count_left = bisect.bisect_right(sorted_coord, median)
    count_right = N - count_left
    ptr_right = count_left
    x = median
    while True:
        x +=1
        sum_x += count_right - count_left
        if sum_x > D:
            break
        # Count number of x_i ==x
        cnt = 0
        while ptr_right < N and sorted_coord[ptr_right] ==x:
            cnt +=1
            ptr_right +=1
        count_left +=cnt
        count_right -=cnt
        list_sum.append(sum_x)
    
    # Iterate to the left
    sum_x = list_sum[0]
    count_left_initial = count_left
    count_right_initial = count_right
    list_sum_left = []
    ptr_left = count_left -1
    x = median
    while True:
        x -=1
        sum_x += count_left - count_right
        if sum_x > D:
            break
        # Count number of x_i ==x
        cnt =0
        while ptr_left >=0 and sorted_coord[ptr_left] ==x:
            cnt +=1
            ptr_left -=1
        count_left -=cnt
        count_right +=cnt
        list_sum_left.append(sum_x)
    
    list_sum.extend(list_sum_left)
    return list_sum

def main():
    import sys
    import threading
    def run():
        N,D = map(int, sys.stdin.readline().split())
        xs = []
        ys = []
        for _ in range(N):
            x,y = map(int, sys.stdin.readline().split())
            xs.append(x)
            ys.append(y)
        sorted_x = sorted(xs)
        sorted_y = sorted(ys)
        list_sum_x = get_sum_list(sorted_x, D)
        list_sum_y = get_sum_list(sorted_y, D)
        list_sum_y_sorted = sorted(list_sum_y)
        ans =0
        for sx in list_sum_x:
            remain = D - sx
            if remain <0:
                continue
            cnt = bisect.bisect_right(list_sum_y_sorted, remain)
            ans += cnt
        print(ans)
    threading.Thread(target=run).start()

if __name__ == "__main__":
    main()