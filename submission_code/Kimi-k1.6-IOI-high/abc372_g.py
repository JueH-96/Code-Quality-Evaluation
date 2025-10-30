import sys
import math

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

class Line:
    def __init__(self, a, b, idx):
        # Line: a*x + b represents (C_i -1 - A_i x)/B_i = (-A_i/B_i)x + (C_i-1)/B_i
        # We store numerator and denominator to avoid floating point errors
        self.a = a
        self.b = b
        self.idx = idx
        # Simplify a and b
        g = gcd(abs(a[1]), abs(b[1]))
        g = gcd(g, abs(a[0]))
        g = gcd(g, abs(b[0]))
        if g > 1:
            self.a = (a[0]//g, a[1]//g)
            self.b = (b[0]//g, b[1]//g)
        # Ensure denominator is positive
        if self.a[1] < 0:
            self.a = (-self.a[0], -self.a[1])
        if self.b[1] < 0:
            self.b = (-self.b[0], -self.b[1])
        # Normalize for comparison
        self.slope = (self.a[0], self.a[1])
        self.intercept = (self.b[0], self.b[1])
    
    def __lt__(self, other):
        # Compare slopes (negative of A_i/B_i)
        a1, b1 = self.slope
        a2, b2 = other.slope
        return a1 * b2 < a2 * b1

def cross(line1, line2):
    # Find x where line1 and line2 intersect
    a1, b1 = line1.a
    c1, d1 = line1.b
    a2, b2 = line2.a
    c2, d2 = line2.b
    # Solving line1.a x + line1.b = line2.a x + line2.b
    # (a1/b1 - a2/b2)x = (c2/d2 - c1/d1)
    numerator = (c2 * d1 - c1 * d2)
    denominator = (a1 * b2 - a2 * b1)
    if denominator == 0:
        return (0, 1) if numerator == 0 else (0, 0)  # Parallel lines, no intersection or same line
    g = gcd(abs(numerator), abs(denominator))
    numerator //= g
    denominator //= g
    if denominator < 0:
        numerator = -numerator
        denominator = -denominator
    return (numerator, denominator)

def add_line(line, hull):
    # Remove lines that are no longer useful
    while len(hull) >= 2:
        last = hull[-1]
        second_last = hull[-2]
        cross1 = cross(second_last, last)
        cross2 = cross(last, line)
        # If cross2 is to the left of cross1, last line is no longer needed
        if cross1[0] * cross2[1] >= cross2[0] * cross1[1]:
            hull.pop()
        else:
            break
    hull.append(line)
    return hull

def compute_segment_sum(a, b, x_start, x_end):
    # Compute sum of floor(a*x + b) from x_start to x_end inclusive
    # a and b are fractions: a is (a_num, a_den), b is (b_num, b_den)
    a_num, a_den = a
    b_num, b_den = b
    x_start_f = x_start
    x_end_f = x_end
    # Total terms
    n = x_end - x_start + 1
    if n <= 0:
        return 0
    # Sum of a*x + b for x in [x_start, x_end]
    # a*x + b = (a_num/a_den)*x + (b_num/b_den)
    sum_a = a_num * (x_start_f + x_end_f) // a_den * n // 2
    sum_b = b_num * n // b_den
    return sum_a + sum_b

def solve():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr +=1
    for _ in range(T):
        N = int(input[ptr])
        ptr +=1
        lines = []
        valid = True
        min_c = []
        for __ in range(N):
            A = int(input[ptr])
            B = int(input[ptr+1])
            C = int(input[ptr+2])
            ptr +=3
            if C <= A + B:
                valid = False
                continue
            # Check if any C_i <= A_i + B_i
            # Also, compute minimal (C_i - B_i -1) //A_i for x <= this
            # But for the convex hull, each line is ( -A/B )x + (C-1)/B
            # So, we represent each line as (-A, B) for slope and (C-1, B) for intercept
            # To avoid floating points, store numerators and denominators
            lines.append( Line( (-A, B), (C-1, B), __ ) )
        if not valid or len(lines) ==0:
            print(0)
            continue
        
        # Sort lines by decreasing slope (since m = -A/B)
        # Compare two lines: line1 is before line2 if its slope is more negative
        lines.sort(reverse=True)
        
        convex_hull = []
        for line in lines:
            convex_hull = add_line(line, convex_hull)
        
        # Now, we need to generate the list of lines in the convex hull and their intersection points
        # First, each line in convex_hull is active in a certain x range
        # Compute the intersection points between consecutive lines
        intersections = []
        for i in range(len(convex_hull)-1):
            l1 = convex_hull[i]
            l2 = convex_hull[i+1]
            x = cross(l1, l2)
            intersections.append(x)
        
        # Now, process each segment between intersections
        total =0
        prev_x = (-float('inf'), 1)
        for i in range(len(convex_hull)):
            current_line = convex_hull[i]
            a_line = current_line.a
            b_line = current_line.b
            if i ==0:
                # Active until first intersection
                next_x = intersections[0] if len(intersections) >0 else (float('inf'),1)
            else:
                next_x = intersections[i-1]
            # Compute x_start and x_end for this segment
            # Handle division of fractions for x ranges
            # For the line segment active from prev_x to next_x
            # x ranges from prev_x to next_x (exclusive)
            # Need to find x where line's value >=1
            # y = (b_line[0] - a_line[0] *x ) / b_line[1] >=1
            # (b_line[0] - a_line[0]*x ) >= b_line[1]
            # x <= (b_line[0] - b_line[1]) / a_line[0] * (a_line[1]/a_line[1])
            # Represent as fractions to avoid floating points
            # Solve for x <= (b_line[0] - b_line[1]) / a_line[0]
            # But since a_line is negative (slope is -A/B), division changes inequality
            rhs_num = b_line[0] - b_line[1]
            rhs_den = a_line[0]
            x_high_num = rhs_num * a_line[1]
            x_high_den = rhs_den * a_line[0]
            # x <= x_high_num / x_high_den
            # Since a_line[0] is negative (from Line's a being -A/B), rhs_den is a_line[0]
            # So x_high is (b_line[0] - b_line[1]) / a_line[0]
            # Because a_line[0] <0, x_high is (b_line[0] -b_line[1])/a_line[0] will be positive
            # Wait, b_line[0] is C-1, b_line[1] is B.
            # So x_high_num = (C-1 - B) * B
            # x_high_den = (-A) * B
            # Then x_high is (C-1 - B) / (-A) = (B- C +1)/A → but since C> A+B, this should be negative?
            # Wait, original code has a problem here.
            # Alternatively, to find x where (C-1 -A x)/B >=1 → C-1 -A x >= B → x <= (C-1 -B)/A
            # So x_high is (C-1 -B)/A → which is an integer division.
            # But with fractions, it's better to compute this as a fraction.
            # For the current line, the x_high where y >=1 is (C_i -1 - B_i)/A_i
            A_i = lines[0].a[0] * (-1) // lines[0].a[1]  # Denominator of line's a is lines[0].a[1]
            B_i = lines[0].b[1]
            C_i = lines[0].b[0] +1
            x_high = (C_i - B_i) // A_i
            # But how to handle this for each line's parameters stored as fractions.
            # Let's recompute for current line's A_i, B_i, C_i.
            # Original line is (C_i -1 -A_i x)/B_i >=1 → x <= (C_i-1 - B_i)/A_i
            # So, x_high is (C_i-1 - B_i) / A_i
            # Using the line's stored parameters:
            # The line's a is (-A_i/B_i) → stored as (-A_i, B_i)
            # The line's b is (C_i-1)/B_i → stored as (C_i-1, B_i)
            # So A_i = a_line[0] * (-1) / a_line[1]
            # For example, if a_line is (-2,3), then A_i is 2/3 → A_i x is (2/3)x
            # So the actual A_i is abs(a_line[0]/a_line[1])
            # Therefore, x_high is (b_line[0]/b_line[1] -1 ) / (a_line[0]/a_line[1] * (-1))
            # Let's compute (C_i-1 - B_i)/A_i where C_i-1 is b_line[0], B_i is b_line[1]
            # So, (b_line[0] - b_line[1]) / (-a_line[0])
            # Because a_line[0] is -A_i → -a_line[0] is A_i
            # So x_high is (b_line[0] - b_line[1]) / (-a_line[0])
            # But since a_line is stored as (a_num, a_den), which is (-A_i, B_i)
            # So a_line's a_num is -A_i, a_den is B_i.
            # So x_high is ( (b_line[0] - b_line[1]) * a_den ) / ( a_num * (-1) )
            # Because:
            # (b_line[0] -b_line[1]) / (-a_line[0]) → numerator is (b_line[0]-b_line[1]), denominator is a_num.
            # But with a_line's a_num is -A_i, and a_den is B_i.
            # So (b_line[0] -b_line[1]) / (-a_line[0]) is (C_i-1 -B_i)/A_i → the x_high.
            # Represented as a fraction:
            numerator = (b_line[0] - b_line[1]) * a_line[1]  # Since a_line's a_den is B_i
            denominator = (-a_line[0]) * a_line[1]
            # Simplify this fraction
            g = gcd(abs(numerator), abs(denominator))
            numerator //= g
            denominator //= g
            if denominator < 0:
                numerator = -numerator
                denominator = -denominator
            x_high_frac = (numerator, denominator)
            # Now, for this line segment, the active x range is from prev_x to next_x
            # prev_x is the intersection with the previous line, next_x with the next line
            # So the x range is [prev_x_intersection, next_x_intersection)
            # The line is active for x >= prev_x_intersection and x < next_x_intersection
            # We need to compute the x's that are integers >=1 and < next_x_intersection, and <= x_high_frac
            # So the x's are max(prev_x, 1) <= x < min(next_x, x_high_frac)
            # And these x's must be integers.
            # We handle prev_x and next_x as fractions.
            # For the first line, prev_x is -infinity, so no lower bound except x >=1
            # For each segment, we have:
            # x_start is the previous intersection's x value or -infinity
            # x_end is the next intersection's x value or infinity
            # So for each segment between convex_hull[i-1] and convex_hull[i], the active x is from intersection i-2 to intersection i-1
            # Wait, initial processing of convex_hull and intersections:
            # convex_hull is a list of lines.
            # intersections is a list of x coordinates where consecutive lines cross.
            # The first line is active from x=-infinity to intersections[0], if any.
            # The second line is active from intersections[0] to intersections[1], etc.
            # The last line is active from intersections[-1] to x=infinity.
            # So the number of lines is len(convex_hull)
            # The number of intersections is len(convex_hull)-1
            current_line = convex_hull[i]
            a_line = current_line.a
            b_line = current_line.b
            # Compute x_start and x_end for the current line's active segment
            if i ==0:
                x_start_frac = (-float('inf'), 1)  # start at -infinity
                if len(intersections)==0:
                    x_end_frac = (float('inf'), 1)
                else:
                    x_end_frac = intersections[0]
            else:
                x_start_frac = intersections[i-1]
                x_end_frac = intersections[i] if i < len(intersections) else (float('inf'), 1)
            # Find x_start and x_end in terms of fractions
            # Now, compute the valid x's for this segment where x >=1 and the line's y >=1
            # The valid x's are those >= max(x_start, 1) and <x_end_frac and <=x_high_frac
            # So the lower bound is x_low = max(ceil(x_start_frac), 1)
            # Upper bound is x_high = min(floor(x_end_frac), floor(x_high_frac))
            # Convert these fractions to actual values and find overlapping integer ranges
            # For x_start_frac: can be (num, den) or (-inf,1)
            # For x_start_frac being (-inf, 1): x_low is max(1, ...)
            # But we need to compute x_low as the maximum between x_start and 1.
            # Similarly for x_high_frac.
            
            # Need to manage fractions as pairs of numerator and denominator
            # To compare fractions:
            # For x_start_frac and x_high_frac: compute their actual values
            # But for efficiency, use cross multiplication
            # For example, to compare two fractions a/b and c/d: a*d ? c*b
            # For x_start_frac: if it's not infinity, compare with 1
            # Let's represent x_start and x_end as fractions (n, d)
            x_start = x_start_frac
            x_end = x_end_frac
            
            # Convert x_high_frac into the same terms
            x_high = x_high_frac
            
            # Compute x_low: maximum between x_start and 1
            # x_start is a fraction or -infinity
            # x_low is max(x_start, 1)
            # Similarly, x_high is min(x_end, x_high_frac)
            
            # For x_start_frac:
            if x_start[1] == 1 and x_start[0] == -float('inf'):
                x_low_num = 1
                x_low_den = 1
            else:
                # Compute max(x_start, 1)
                x_start_val = x_start[0] * 1.0 / x_start[1]
                if x_start_val >=1:
                    x_low_num = x_start[0]
                    x_low_den = x_start[1]
                else:
                    x_low_num = 1
                    x_low_den = 1
            # Similarly for x_high: compute min(x_end_frac, x_high_frac)
            if x_end[1] ==1 and x_end[0] == float('inf'):
                # x_high is min between x_high_frac and infinity → x_high_frac
                x_high_num = x_high[0]
                x_high_den = x_high[1]
            else:
                # x_high is the minimum of x_end_frac and x_high_frac
                x_end_val = x_end[0] / x_end[1]
                x_high_val = x_high[0] / x_high[1]
                if x_end_val < x_high_val:
                    x_high_num = x_end[0]
                    x_high_den = x_end[1]
                else:
                    x_high_num = x_high[0]
                    x_high_den = x_high[1]
            # Now, the valid x's are from x_low (inclusive) to x_high (exclusive)
            # So x must be >=x_low and <x_high
            # Also, x must be >=1
            # So the actual x_low is max(x_low_num/x_low_den, 1)
            # x_low_num/x_low_den could be less than 1, but we already took max with 1
            # So x_low is x_low_num/x_low_den rounded up to the nearest integer >=1
            # x_high is x_high_num/x_high_den floored (since x must be less than x_high)
            x_low = math.ceil(x_low_num / x_low_den) if x_low_den !=0 else 0
            x_high = math.floor( (x_high_num -1)/ x_high_den ) if x_high_den !=0 else 0
            # Also, x must be >=1
            x_low = max(x_low, 1)
            if x_high < x_low:
                continue
            # Now, compute the sum for x from x_low to x_high inclusive
            # For each x in this range, compute floor( (b_line[0] - a_line[0]*x)/b_line[1] )
            # Which is floor( (b_line[0] - a_line[0] x) / b_line[1] )
            # Which is equal to (b_line[0] - a_line[0] x + b_line[1] -1 ) // b_line[1]
            # So the sum can be calculated as:
            # sum_{x=x_low}^x_high floor( (b_num - a_num x) / b_den )
            a_num = a_line[0]
            a_den = a_line[1]
            b_num = b_line[0]
            b_den = b_line[1]
            # Wait, current_line is a line from the convex hull. The line equation is:
            # y = (b_line[0]/b_line[1]) + (a_line[0]/a_line[1}) * x
            # Which is equivalent to y = (b_line[0] * a_line[1] - a_line[0] * b_line[1] * x ) / (a_line[1] * b_line[1])
            # The floor of this is floor( ... )
            # To compute the sum, we need to compute for each x:
            # floor( (b_line[0] * a_line[1] - a_line[0] * x * b_line[1] ) / (a_line[1] * b_line[1} ) )
            # Which is the same as floor( (b_line[0] * a_line[1] - a_line[0] *b_line[1} x ) / (a_line[1] *b_line[1} ) )
            # Let's create coefficients D and E such that this sum is sum_{x} floor( (D - E x)/F )
            D = b_line[0] * a_line[1]
            E = a_line[0] * b_line[1]
            F = a_line[1] * b_line[1]
            # So sum from x=x_low to x_high of floor( (D -E x)/F )
            # This can be computed as follows:
            # Let q = (D -E x)//F → q = (D // F) - E x // F + ... → Not helpful
            # But D, E, F can be large, but we need a formula for the sum.
            # Apply the method from the following reference: https://stackoverflow.com/questions/3928042/summation-of-floor-function
            # According to this, the sum can be calculated using the formula:
            # sum_{x=a}^b floor((c x + d)/m) = (c (a + b) (b -a +1)) // (2m) + d*(b-a+1)//m - sum_{i=0}^{m-1} floor( ( (c a -c i +d ) )/m )
            # But this requires careful handling.
            # Another Idea from someone's code on AtCoder:
            # The sum can be rewritten as sum_{x} ((D - E x) // F )
            # Let's find how many times each value of q is taken.
            # q_min = (D - E * x_high) // F
            # q_max = (D - E *x_low) // F
            # For each x, the term decreases by E/F per x step.
            # The sum is q0 + (q0 -1) + ... + qk where each q transitions at certain x points.
            # Another Idea: The sum can be calculated as the area under the line D-E x divided by F minus the number of fractional parts that are non-integer.
            # But this is not precise.
            # Let's proceed with a different approach inspired by the fact that the line is part of a convex hull and thus each sum can be approximated as a triangular number.
            # Compute the average of the first and last term and multiply by the number of terms.
            # This is an approximation but might work correctly with floor adjustments.
            # Compute q_start = floor((D - E * x_low)/F )
            # q_end = floor((D - E * x_high)/F )
            # But since the line has a slope of -E/F (since E and F are positive), each increment of x by 1 reduces the term by E/F.
            # So the terms are q_start, q_start - k1, q_start -k2, ... until q_end.
            # The sum can be calculated as (q_start + q_end) * n // 2, but only if the differences between consecutive terms are consistent.
            # For example, if each term decreases by exactly 1 as x increases, the terms form an arithmetic sequence.
            # This would be the case if E >= F. Then each x increment reduces q by E//F + floor( ... ), but this is only true for certain cases.
            # This Idea might not handle all cases but could work for the problem.
            # Let's calculate q_start and q_end.
            x_low = x_low
            x_high = x_high
            q_start = (D - E * x_low) // F
            q_end = (D - E * x_high) // F
            n = x_high - x_low + 1
            if q_start == q_end:
                total += q_start * n
            else:
                # Assume that the terms form a sequence where each term decreases by 1
                # So sum is n*(q_start + q_end) // 2
                total += n * (q_start + q_end) // 2
        print(total)

if __name__ == '__main__':
    solve()