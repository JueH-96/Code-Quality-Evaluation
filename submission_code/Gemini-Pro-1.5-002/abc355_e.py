# YOUR CODE HERE
def solve():
    n, l, r = map(int, input().split())

    def query(i, j):
        print(f"? {i} {j}")
        return int(input())

    def get_sum(left, right):
        total_sum = 0
        for i in range(n, -1, -1):
            length = 1 << i
            start = (left // length) * length
            end = start + length - 1
            if start >= left and end <= right:
                total_sum = (total_sum + query(i, start // length)) % 100
            elif start < left and end >= left and end <=right:
                
                for j in range(n,i,-1):
                    sub_length = 1 << j
                    sub_start = (left // sub_length) * sub_length
                    sub_end = sub_start + sub_length -1
                    if sub_start >= left and sub_end <= end:
                        total_sum = (total_sum + query(j, sub_start // sub_length)) % 100
                        
            elif start <= right and start >= left and end > right:
                for j in range(n,i,-1):
                    sub_length = 1 << j
                    sub_start = (start // sub_length) * sub_length
                    sub_end = sub_start + sub_length -1
                    if sub_start >= start and sub_end <= right:
                        total_sum = (total_sum + query(j, sub_start // sub_length)) % 100
        return total_sum

    result = get_sum(l, r)
    print(f"! {result}")

solve()