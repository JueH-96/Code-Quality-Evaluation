# YOUR CODE HERE
def solve():
    n = int(input())
    
    def to_good_integer(k):
        res = ""
        while k > 0:
            res = str(2 * (k % 5)) + res
            k //= 5
        return res if res else "0"

    print(to_good_integer(n))

solve()