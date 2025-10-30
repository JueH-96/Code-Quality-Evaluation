import sys

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        n = int(data[index]); index += 1
        k = int(data[index]); index += 1
        a = list(map(int, data[index:index+n]))
        index += n
        
        if k == 2:
            found_even = False
            for num in a:
                if num % 2 == 0:
                    found_even = True
                    break
            if found_even:
                results.append("0")
            else:
                results.append("1")
                
        elif k == 3:
            min_ops = 10
            for num in a:
                r = num % 3
                if r == 0:
                    min_ops = 0
                    break
                else:
                    steps = 3 - r
                    if steps < min_ops:
                        min_ops = steps
            results.append(str(min_ops))
            
        elif k == 4:
            even_count = 0
            count4 = 0
            min_steps_4 = 10**6
            for num in a:
                if num % 2 == 0:
                    even_count += 1
                if num % 4 == 0:
                    count4 += 1
                r = num % 4
                if r == 0:
                    step_i = 0
                else:
                    step_i = 4 - r
                if step_i < min_steps_4:
                    min_steps_4 = step_i
                    
            if count4 >= 1 or even_count >= 2:
                results.append("0")
            else:
                if even_count == 1:
                    op_b = 1
                else:
                    op_b = 2
                ans = min(min_steps_4, op_b)
                results.append(str(ans))
                
        elif k == 5:
            min_ops = 10
            for num in a:
                r = num % 5
                if r == 0:
                    min_ops = 0
                    break
                else:
                    steps = 5 - r
                    if steps < min_ops:
                        min_ops = steps
            results.append(str(min_ops))
            
    print("
".join(results))

if __name__ == "__main__":
    main()