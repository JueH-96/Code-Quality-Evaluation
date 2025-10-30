import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    t = int(input[ptr])
    ptr += 1
    for _ in range(t):
        n = int(input[ptr])
        ptr += 1
        a = list(map(int, input[ptr:ptr+n]))
        ptr += n
        global_max = -float('inf')
        prev_even = -float('inf')
        prev_odd = -float('inf')
        for num in a:
            current_parity = num % 2
            if current_parity == 0:
                new_even = num
                if prev_odd != -float('inf'):
                    new_even = max(new_even, prev_odd + num)
                new_odd = -float('inf')
            else:
                new_odd = num
                if prev_even != -float('inf'):
                    new_odd = max(new_odd, prev_even + num)
                new_even = -float('inf')
            current_max = max(new_even, new_odd)
            if current_max > global_max:
                global_max = current_max
            prev_even, prev_odd = new_even, new_odd
        print(global_max)

if __name__ == "__main__":
    main()