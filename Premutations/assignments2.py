from Premutations.perm import *

"""Exercise 1: Permutations"""

def prem_1():
    print("Create a test permutation of length 20")
    q = test_permutation(20)
    print(q)
    print("\nDefine a permutation of length 10")
    p = [1, 2, 3, 0, 5, 4, 6, 7, 8, 9]
    print(p)
    print_permutation(p)
    print(is_trivial(p))
    print(p[0])
    print(cycles(p))
    print("\nCreate a trivial permutation of length 10")
    r = trivial_permutation(10)
    print(r)
    print_permutation(r)
    print(r[0])
    print(is_trivial(r))
    print("\nPremutations from cycles")
    p = permutation_from_cycles(10, [[0, 1, 2, 3], [4, 5]])
    print(p)
    print_permutation(p)


"""Exercise 2:Implement a function composition with two arguments p and q for computing the
composition of the two permutations p and q (represented by p◦q)."""

def composition(p, q):
    if len(p) != len(q):
        print("cannot compose permutations of varying size")
        return -1
    else:
        res = [0] * len(p)
        for i in range(0, len(p)):
            res[i] = p[q[i]]
    return res

def test_composition():
    print("\nTesting composition: ")
    p = [1, 2, 3, 0, 5, 6, 4, 8, 7]
    print_permutation(p)
    q = composition(p, p)
    print_permutation(q)


"""Exercise 3:Implement  a  function inverse with  an  argument p for  computing  the inverse of a permutation p. """

def inverse(p):
    res = [0]*len(p)
    for i in range(0, len(p)):
        res[p[i]] = p[i]
    return res

def test_inverse():
    print("\nTesting inverse: ")
    p = [1, 2, 3, 0, 5, 6, 4, 8, 7]
    print_permutation(p)
    q = inverse(p)
    print(q)
    print_permutation(q)

"""Exercise 4:Implement a method power with two arguments p and i, which can be used to compute the ith power p^i of a 
permutation p,  where i is an integer:  typing power(p,5) should give p^5 = p◦p◦p◦p◦p.
Ensure that the method also works for i = 0 and i < 0."""

def power_slow(p, i):
    res = []
    for j in range(0, len(p)):
        res.append(j)
    non_neg = abs(i) == i
    for k in range(0, abs(i)):
        if non_neg:
            res = composition(p, res)
        else:
            res = composition(inverse(p), res)
    return res

def bit_length(n):
    bits = 0
    while n >> bits:
        bits += 1
    return bits

def power(p, i):
    res = []
    for j in range(0, len(p)):
        res.append(j)
    ans = res
    non_neg = abs(i) == i
    if non_neg:
        first = True
        while i != 0:
            if first:
                res = p
                if i & 1:
                    ans = res
                first = False
            else:
                res = composition(res, res)
                i = i >> 1
                if i & 1:
                    ans = composition(res, ans)
    else:
        for k in range(0, abs(i)):
            res = composition(inverse(p), res)
        ans = res
    return ans

def test_power():
    print("\nTesting power: ")
    p = [1, 2, 3, 0, 5, 6, 4, 8, 7]
    q = composition(p, p)
    r = power(p, 2)
    print("composition p,p eqauls power(p,2) :", q == r)
    print("power(p,1) equals p               :", power(p, 1) == p)
    print("power(p,0) is trivial             :", is_trivial(power(p, 0)))
    s = power(p, -1)
    t = inverse(p)
    print("power(0, -1) equals inverse(p)    :", s == t)
    # z = test_permutation(15)
    # print("power_slow() equals power() fast: ", power_slow(z, period(z)) == power(z, period(z)))
    zz = test_permutation(200)
    print_permutation(power(zz, period(zz)))




"""Exercise 5:Write a function period, with one argument p, that computes the smallest integer i ≥ 1 such that p^i is the trivial permutation.
  Verify the correctness of the answer."""


def gcd(a, b):
    a, b = abs(a), abs(b)
    while b != 0:
        a, b = b, a % b
    return a


def lcm(a, b):
    return a*b // gcd(a, b)


def period_slow(p):
    q = p
    count = 1
    while not is_trivial(q):
        q = composition(p, q)
        print_permutation(q)
        count += 1
    return count


def period(p):
    n = 1
    cyc = cycles(p)
    for c in cyc:
        n = lcm(n, len(c))
    return n


def test_period():
    p = [1, 2, 3, 0, 5, 6, 4, 8, 7]
    print("period_slow equals period: ", period_slow(p) == period(p))
    q = test_permutation(100)
    print("p has a period of: ", period(q))
    r = test_permutation(250)
    print("p has a period of: ", period(r))






if __name__ == '__main__':
    # prem_1()
    # test_composition()
    # test_inverse()
    # test_power()
    # test_period()
    p = test_permutation(20)
    print_permutation(p)
    print(period_slow(p))
    # print_permutation(p)
