# YOUR CODE HERE
def print_carpet(n):
    def helper(k, i_start, i_end, j_start, j_end):
        if k == 0:
            return ['#' * (j_end - j_start)] * (i_end - i_start)
        half = 3 ** (k - 1)
        result = []
        for i in range(i_start, i_end):
            if i_start + half <= i < i_start + 2 * half:
                result.append('#' * (j_start) + '.' * (j_end - j_start))
            else:
                result.append(helper(k - 1, i, i + 1, j_start, j_start + half) +
                              helper(k - 1, i, i + 1, j_start + 2 * half, j_end))
        return result
    
    return [''.join(row) for row in helper(n, 0, 3**n, 0, 3**n)]

n = int(input())
for row in print_carpet(n):
    print(row)