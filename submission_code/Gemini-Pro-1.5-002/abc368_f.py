# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))

    def get_divisors(num):
        divisors = []
        for i in range(1, int(num**0.5) + 1):
            if num % i == 0:
                divisors.append(i)
                if i * i != num:
                    divisors.append(num // i)
        return divisors

    def winner(arr):
        xor_sum = 0
        for num in arr:
            divisors = get_divisors(num)
            grundy = 0
            for div in divisors:
                if div != num:
                    
                    temp_arr = list(arr)
                    temp_arr[arr.index(num)] = div
                    
                    grundy |= (1 << winner(temp_arr))
            
            
            mex = 0
            while (grundy >> mex) & 1:
                mex += 1
            xor_sum ^= mex
        
        return xor_sum


    result = winner(a)
    if result == 0:
        print("Bruno")
    else:
        print("Anna")

solve()