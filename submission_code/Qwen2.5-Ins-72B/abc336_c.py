# YOUR CODE HERE
def find_nth_good_integer(N):
    good_digits = ['0', '2', '4', '6', '8']
    good_integers = []
    
    def generate_good_integers(current, length):
        if length == 0:
            if current != '0':
                good_integers.append(int(current))
            return
        for digit in good_digits:
            generate_good_integers(current + digit, length - 1)
    
    max_length = 1
    while len(good_integers) < N:
        generate_good_integers('', max_length)
        max_length += 1
    
    good_integers.sort()
    return good_integers[N - 1]

N = int(input())
print(find_nth_good_integer(N))