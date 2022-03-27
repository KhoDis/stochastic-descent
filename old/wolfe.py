import numpy as np
from deprecated import deprecated

from old.function import grad, f


@deprecated
def draw_wolfe_conditions(point, direction, old_lr):
    def first(alpha):
        return first_wolfe_condition(point, direction, alpha)

    def second(alpha):
        return second_wolfe_condition(point, direction, alpha)

    steps = 100
    left_bound, right_bound = 0, 1
    arr = np.linspace(left_bound, right_bound, steps)

    new_lr = line_search(point, direction, old_lr)

    line = "".join(list(map(lambda x: "-" if first(x) and second(x) else " ", arr)))
    line_arr = [char for char in line]

    def get_point(lr):
        return int(left_bound + lr * steps / right_bound)

    old_point = get_point(old_lr)
    new_point = get_point(new_lr)

    line_arr[old_point] = 'O'
    line_arr[new_point] = 'N'

    line = "".join(line_arr)

    print(f'|{line}|')
    print('old', first(old_lr), second(old_lr), old_lr)
    print('new', first(new_lr), second(new_lr), new_lr)


@deprecated
def line_search(point, direction, slr):
    def first(alpha):
        return first_wolfe_condition(point, direction, alpha)

    def second(alpha):
        return second_wolfe_condition(point, direction, alpha)

    def search(ll, rr):
        mid = (ll + rr) / 2
        while not (first(mid) and second(mid)):
            if first(mid) and not second(mid):
                ll = mid
            else:
                rr = mid
            mid = (ll + rr) / 2
        return mid

    def find_bound(search_lambda, cond1, cond2, start, reverse_factor):
        bound = start
        factor = 0.5
        while not cond2(bound):
            bound += reverse_factor * factor
            factor *= 2
        if not cond1(bound):
            return search_lambda(bound, start)
        else:
            return bound

    first_lr = first(slr)
    second_lr = second(slr)
    if first_lr and not second_lr:
        return find_bound(lambda right, lr: search(lr, right), first, second, slr, 1)
    elif not first_lr and second_lr:
        return find_bound(lambda left, lr: search(left, lr), second, first, slr, -1)
    elif first_lr and second_lr:
        return slr
    else:
        return None


@deprecated
def first_wolfe_condition(point, direction, alpha):
    beta = 0.25  # less -> more possibilities for alpha
    next_point = point + alpha * direction
    return f(*next_point) <= f(*point) + beta * alpha * get_gamma(direction, point)


@deprecated
def second_wolfe_condition(point, direction, alpha):
    beta = 0.25  # less -> closer to the minimum
    next_point = point + alpha * direction
    return get_gamma(direction, next_point) >= beta * get_gamma(direction, point)


@deprecated
def get_gamma(direction, point):
    """gamma === (∇f(x))ᵀ * -∇f(x)"""
    return np.matmul(np.matrix(direction), np.transpose(np.matrix(np.array(grad(point))))).item()
