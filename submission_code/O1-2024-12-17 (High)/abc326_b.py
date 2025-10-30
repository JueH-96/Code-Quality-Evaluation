# YOUR CODE HERE
def main():
    import sys
    
    def is_326_like(num):
        s = str(num)
        h, t, o = map(int, s)  # hundreds, tens, ones
        return h * t == o
    
    N = int(sys.stdin.readline().strip())
    for x in range(N, 1000):
        if is_326_like(x):
            print(x)
            break

# Don't forget to call main()
main()