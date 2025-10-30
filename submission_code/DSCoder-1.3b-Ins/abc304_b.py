# YOUR CODE HERE

N = int(input())

if N <= 10**3-1:
    print(N)
elif N >= 10**4 and N <= 10**5-1:
    print(int(N/10))
elif N >= 10**5 and N <= 10**6-1:
    print(int(N/100))
elif N >= 10**6 and N <= 10**7-1:
    print(int(N/1000))
elif N >= 10**7 and N <= 10**8-1:
    print(int(N/10000))
elif N >= 10**8 and N <= 10**9-1:
    print(int(N/100000))