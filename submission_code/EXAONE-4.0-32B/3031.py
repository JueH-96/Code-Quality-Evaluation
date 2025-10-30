MOD = 12345
primes = [3, 5, 823]

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        m = len(grid[0])
        
        exp3 = [[0] * m for _ in range(n)]
        exp5 = [[0] * m for _ in range(n)]
        exp823 = [[0] * m for _ in range(n)]
        coprime_part_grid = [[1] * m for _ in range(n)]
        
        zero_count = 0
        total_exponents = [0, 0, 0]
        total_coprime = 1
        
        for i in range(n):
            for j in range(m):
                x = grid[i][j]
                if x == 0:
                    zero_count += 1
                else:
                    temp = x
                    exp_list = [0, 0, 0]
                    for idx, p in enumerate(primes):
                        while temp % p == 0:
                            exp_list[idx] += 1
                            temp //= p
                    exp3[i][j] = exp_list[0]
                    exp5[i][j] = exp_list[1]
                    exp823[i][j] = exp_list[2]
                    coprime_part_grid[i][j] = temp
                    total_exponents[0] += exp_list[0]
                    total_exponents[1] += exp_list[1]
                    total_exponents[2] += exp_list[2]
                    total_coprime = (total_coprime * temp) % MOD
        
        if zero_count >= 2:
            return [[0] * m for _ in range(n)]
        
        elif zero_count == 1:
            P = total_coprime
            for idx, p in enumerate(primes):
                P = (P * pow(p, total_exponents[idx], MOD)) % MOD
            res = [[0] * m for _ in range(n)]
            for i in range(n):
                for j in range(m):
                    if grid[i][j] == 0:
                        res[i][j] = P
            return res
        
        else:
            res = [[0] * m for _ in range(n)]
            for i in range(n):
                for j in range(m):
                    a = exp3[i][j]
                    b = exp5[i][j]
                    c = exp823[i][j]
                    temp_val = coprime_part_grid[i][j]
                    inv_temp = pow(temp_val, -1, MOD)
                    new_coprime = (total_coprime * inv_temp) % MOD
                    new_exp3 = total_exponents[0] - a
                    new_exp5 = total_exponents[1] - b
                    new_exp823 = total_exponents[2] - c
                    power_val = pow(3, new_exp3, MOD)
                    power_val = (power_val * pow(5, new_exp5, MOD)) % MOD
                    power_val = (power_val * pow(823, new_exp823, MOD)) % MOD
                    res[i][j] = (new_coprime * power_val) % MOD
            return res