# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import math
    t = int(sys.stdin.readline())
    for _ in range(t):
        n,k = map(int, sys.stdin.readline().split())
        a = list(map(int, sys.stdin.readline().split()))
        cost = None
        if k == 2:
            has_even = any(x%2==0 for x in a)
            if has_even:
                print(0)
            else:
                print(1)
        elif k == 3:
            min_cost = 2  # Max possible cost is 2
            for x in a:
                if x%3==0:
                    min_cost = 0
                    break
                else:
                    c = (3 - x%3)%3
                    if c < min_cost:
                        min_cost = c
            print(min_cost)
        elif k==5:
            min_cost = 4  # Max possible cost is 4
            for x in a:
                if x%5==0:
                    min_cost = 0
                    break
                else:
                    c = (5 - x%5)%5
                    if c < min_cost:
                        min_cost = c
            print(min_cost)
        elif k==4:
            count_div4 = sum(1 for x in a if x%4==0)
            if count_div4 >=1:
                print(0)
            else:
                count_even = sum(1 for x in a if x%2==0)
                if count_even >=2:
                    print(0)
                else:
                    # Option 1: Make any a_i divisible by 4
                    min_cost = 3  # Max possible cost is 3
                    for x in a:
                        c = (4 - x%4)%4
                        if c == 0:
                            min_cost = 0
                            break
                        elif c < min_cost:
                            min_cost = c
                    # Option 2: Make numbers even until count_even>=2
                    cost_even = max(0, 2 - count_even)*1
                    total_cost = min(min_cost, cost_even)
                    print(total_cost)
        else:
            print(0)  # Should not reach here, k in [2,5]
threading.Thread(target=main).start()