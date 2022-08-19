from pickle import PicklingError
from typing import Callable, List

import pathos.multiprocessing as mp
from tqdm import tqdm


def try_parallelization(
    executable: Callable,
    params_list: List,
    max_jobs: int = 4,
    multiple_params: bool = False,
    disable_tqdm: bool = True,
):
    """
    Runs n jobs in parallel by calling an executable for each parameter given in the parameter list
    """
    results = []

    # execute jobs sequentially when max_jobs is smaller than 2 to avoid getting
    # errors due to multiprocessing library
    if max_jobs < 2:
        for params in tqdm(params_list, disable=disable_tqdm):
            # unpack parameters if multiple_params is true
            if multiple_params:
                result = executable(*params)
            else:
                result = executable(params)
            results.append(result)

    else:
        try:
            if max_jobs < 2:
                raise "Warning, max jobs is smaller than 2. Will run it sequentially"

            # use starmap to unpack multiple parameters
            if multiple_params:
                with mp.Pool(max_jobs) as pool:
                    results = list(
                        pool.starmap(executable, tqdm(params_list, disable=disable_tqdm))
                    )
            else:
                with mp.Pool(max_jobs) as pool:
                    results = list(pool.map(executable, tqdm(params_list, disable=disable_tqdm)))

        except PicklingError:
            print(
                """Could not pickle function. If you are running your code from a notebook try to restart and clear its output.
                If that does not work, make sure the passed function can be serialized by pathos.multiprocessing.
                Error:""",
                PicklingError,
            )
            raise PicklingError

    return results
