import numpy as np

def sine_function(x, a, b):
    x_rad = np.deg2rad(x)
    return a * np.sin(x_rad) + b

def chi_squared(x, y, y_error, parameters_optimised, function):
  """
  Calculate the chi-squared value for a given dataset and model.

  Parameters:
  x (list or array):                    Independent variable data.
  y (list or array):                    Independent variable data.
  y_error (list or array):              Independent variable data error.
  parameters_optimised (list or tuple): The optimised parameters for the model function.
  function (callable):                  The model function to fit the data. This should take 'x' as its first argument.
                                        and accept 'parameters_optimised' as additional arguments.

  Returns:
  float: The chi-squared value, which quantifies the degree to which the data fits the model function. Lower values indicate a better fit.
  """
  return np.sum(((y - function(x, *parameters_optimised)) / y_error) **2)

def assign_morph(e_probs, s_probs):
    e_class = []
    s_class = []
    for e, s in zip(e_probs, s_probs):
        if e > s:
            e_class.append(1)
            s_class.append(0)
        elif s > e:
            e_class.append(0)
            s_class.append(1)
        else:
            #Tiebreaker: if the probabilities are equal, classify as unknown
            e_class.append(0)
            s_class.append(0)
    return e_class, s_class

def calculate_theta(bcg_ra, bcg_dec, gal_ra, gal_dec):
    """
    Compute the angle (theta) in radians between a BCG and satellite galaxy.
    """
    if (isinstance(gal_ra, str) and gal_ra.strip() == "[]") or (isinstance(gal_dec, str) and gal_dec.strip() == "[]"):
        return []
    gal_ra = np.array(gal_ra, dtype=float)
    gal_dec = np.array(gal_dec, dtype=float)
    avg_dec = np.radians((bcg_dec + gal_dec)/2)
    delta_ra = np.radians(bcg_ra - gal_ra)*np.cos(avg_dec)
    delta_dec = np.radians(bcg_dec - gal_dec)
    theta_raw = np.arctan2(delta_ra, delta_dec)
    theta_clockwise = (2*np.pi - (theta_raw + np.pi)) % (2 * np.pi)
    return np.degrees(theta_clockwise)