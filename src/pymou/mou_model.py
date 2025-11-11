#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2019 Matthieu Gilson, Andrea Insabato, Gorka Zamora-López

Released under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

Main class to deal with construction, simulation and estimation for the
multivariate Ornstein-Uhlenbeck (MOU) process.
"""
# TODO: Revise all the security checks at the beginning of each function/class.
# Surely we can improve and simplify these.
#    - Maybe add an io_helper module like in SiReNetA, or anything more elegant.
#    - avoid use of isscalar() because complex numbers are also scalar.
#    - avoid using type(lag) == int
#    - Check better isinstance(x, numbers.Number), isinstance(x, int),
#      isinstance(x, np.integer), isinstance(x, float), isinstance(x, np.floating)

# Standard library imports
import warnings
# Third-party imports
import numpy as np
import scipy.linalg as spl
import scipy.stats as stt
from sklearn.base import BaseEstimator


###############################################################################
class MOU(BaseEstimator):
    """
    Description of the class and a summary of its parameters, attributes and
    methods.

    Parameters
    ----------
    n : int
        Number of nodes in the network.
    J : ndarray (2d) of shape (n,n)
        Jacobian matrix between the nodes. The diagonal corresponds to a vector
        of time constants. For off-diagonal elements, the first dimension
        corresponds to target nodes and the second dimension to source nodes
        (J_ij is from i to j).
    mu : ndarray (1d) of length n
        Mean vector of the inputs to the nodes.
    Sigma : ndarray (2d) of shape (n,n)
        Covariance matrix of the inputs to the nodes (multivariate Wiener
        process).

    Methods
    -------
    calc_Q0_from_param : Calculates the zero-lag covariance matrix from a Jacobian J and an
        input covariance matrix Sigma

    calc_Qlag_from_param : Calculates the zero-lag covariance matrix from a Jacobian J and an
        input covariance matrix Sigma

    fit : Fit the model to a time series (time x nodes). If existing, the
        previous parameters (connectivity, etc.) are erased and replaced.

    fit_LO : Fit method relying on Lyapunov optimization (gradient descent).

    fit_moments : Fit method with maximum likelihood.

    score : Returns the goodness of fit after the optimization.

    simulate : Simulate the activity of the MOU process determined by J, mu and
        Sigma.
    """

    def __init__(self, C=None, tau=1.0, mu=0.0, Sigma=None,
                random_state=None):
        """Initialize self. See help(MOU) for further information.
        The reason for separating the diagonal and off-diagonal elements in
        the Jacobian comes from focusing on the connectivity matrix as a graph.
        """

        # SECURITY CHECKS AND ARRANGEMENTS FOR THE PARAMETERS
        # Construct Jacobian
        if C is None:
            # 10 nodes by default
            self.n = 10
            # unconnected network
            C_tmp = np.zeros([self.n, self.n], dtype=np.float64)
        elif type(C) == np.ndarray:
            if (not C.ndim == 2) or (not C.shape[0] == C.shape[1]):
                raise TypeError("""Argument C in MOU constructor must be square
                    matrix (2D).""")
            self.n = C.shape[0]
            C_tmp = C
        else:
            raise TypeError("""Only matrix accepted for argument C in MOU
                constructor.""")

        if np.isscalar(tau):
            if tau <= 0:
                raise ValueError( """Scalar argument tau in MOU constructor
                    must be negative for stability.""" )
            else:
                tau_tmp = np.ones(self.n) * tau
        elif type(tau) == np.ndarray:
            if (not tau.ndim == 1) or (not tau.shape[0] == self.n):
                raise TypeError("""Vector argument tau in MOU constructor must
                    be of same size as diagonal of C.""")
            tau_tmp = np.copy(tau)
        else:
            raise TypeError("""Only scalar value or vector accepted for argument
                tau in MOU constructor.""")

        self.J = -np.eye(self.n) / tau_tmp + C_tmp
        if np.any(np.linalg.eigvals(self.J)>0):
            warnings.warn("""The constructed MOU process has a Jacobian with negative
                  eigenvalues, corresponding to unstable dynamics.""",
                  category=RuntimeWarning )

        # Inputs
        if np.isscalar(mu):
            self.mu = mu
        elif type(mu) == np.ndarray:
            if (not mu.ndim == 1) or (not mu.shape[0] == self.n):
                raise TypeError("""Vector argument mu in MOU constructor must be
                    of same size as diagonal of C.""")
            self.mu = mu
        else:
            raise TypeError("""Only scalar value or vector accepted for argument
                tau in MOU constructor.""")

        if Sigma is None:
            # uniform unit variance by default
            self.Sigma = np.eye(self.n, dtype=np.float64)
        elif np.isscalar(Sigma):
            if not (Sigma>0):
                raise TypeError("""Scalar argument Sigma in MOU constructor must
                    be non-negative (akin to variance).""")
            self.Sigma = np.eye(self.n, dtype=np.float64)
        elif type(Sigma) == np.ndarray:
            if (not Sigma.ndim == 2) or (not Sigma.shape[0] == Sigma.shape[1]) \
                                   or (not Sigma.shape[0] == self.n):
                raise TypeError("""Matrix argument Sigma in MOU constructor must
                    be square and of same size as C.""")
            if (not np.all(Sigma == Sigma.T)) or np.any(np.linalg.eigvals(Sigma) < 0):
                raise ValueError("""Matrix argument Sigma in MOU constructor must
                    be positive semidefinite (hence symmetric).""")
            self.Sigma = Sigma
        else:
            raise TypeError("""Only scalar value or matrix accepted for argument
                Sigma in MOU constructor.""")

        # Set seed for reproducibility
        if random_state is not None:
            np.random.seed(random_state)

        # Create a dictionary to store the diagnostics of fit
        self.d_fit = dict()
        self.d_fit['status'] = 'not fitted'


    # METHODS FOR CLASS MOU ###################################################

    def calc_Q0_from_param(self, J, Sigma):
        """
        Calculates the zero-lag covariance matrix from a Jacobian J and an
        input covariance matrix Sigma

        Parameters
        ----------
        J : ndarray (2d) of shape (n,n)
            The Jacobian matrix.
        Sigma : ndarray (2d) of shape (n,n)
            The input covariance matrix.

        Returns
        -------
        Q0 : ndarray (2d) of shape (n,n)
            The theoretical zero-lag covariance matrix.
        """
        Q0 = spl.solve_continuous_lyapunov(J.T, -Sigma)
        return Q0


    def calc_Qlag_from_param(self, Q0, J, lag):
        """
        Method to calculate  the adequate specific method.

        Parameters
        ----------
        Q0 : ndarray (2d) of shape(n,n)
            The zero-lag covariance matrix.
        J : ndarray (2d) of shape (n,n)
            The Jacobian matrix.
        lag : float
            The covariance lag

        Returns
        -------
        Qlag : ndarray (2d) of shape (n,n)2
            The theoretical covariance matrix with lag.
        """
        Qlag = np.dot( Q0, spl.expm( J * lag ) )
        return Qlag


    def calc_emp_Q(self, X, lag, center=True):
        # TODO: move to utils?
        """
        Calculate the covariance matrix with zero lag and with lag from
        time series X of shape (time x nodes)
        """
        T = X.shape[0]
        # remove mean in the time series if center==True
        if center:
            X = X - np.outer(np.ones(T), X.mean(axis=0))
        # calculate the two covariance matrices with same number of points
        Q0 = np.tensordot(X[0:T-lag,:], X[0:T-lag,:], axes=(0,0))
        Qlag = np.tensordot(X[0:T-lag,:], X[lag:T,:], axes=(0,0))
        # rescale by number of time points -1 (not necessary)
        Q0 /= T - lag - 1
        Qlag /= T - lag - 1
        return Q0, Qlag


    def fit(self, X, y=None, lag=1, method='lyapunov', center=True, **kwargs):
        """
        Wrapper for model fiting from time series to call the specific method.

        Parameters
        ----------
        X : ndarray (2d) of shape (T,n)
            The timeseries data of the system to estimate, formatted as
            time points x nodes (e.g. ROIs).
        y : (for compatibility, not used here), optional.
        lag : int, optional, default: 1
            Lag in time steps (1st dimension of X).
        method : string, optional, default: 'lyapunov'
            Set the optimization method; should be 'lyapunov', 'moments' or
            'spectral'.

        Returns
        -------
        J : ndarray (2d) of shape (self.n,self.n)
            The estimated Jacobian (with number of nodes self.n=n from X).
        Sigma : ndarray (2d) of shape (self.n,self.n)
            Estimated input noise covariance.
        d_fit : dictionary
            A dictionary with diagnostics of the fit. Important keys are:
            'iterations', 'distance' (model error), 'correlation' (goodness of
            fit as measured by Pearson correlation on vectorized matrices).
        """

        # TODO: automatize array checks and dimensionality
        if (not type(X) == np.ndarray) or (not X.ndim == 2):
            raise TypeError("""Argument X must be matrix (time x nodes).""")
        self.n = X.shape[1]

        # call adequate method for optimization and check for specific arguments
        if method == 'lyapunov' or 'moments':
            if (not type(lag) == int) or (lag <= 0):
                raise ValueError('Scalar value lag must an integer >0')
            # calculate bjective covariance matrices (empirical)
            Q0_obj, Q1_obj = self.calc_emp_Q(X, lag)
            if method == 'lyapunov':
                return self.fit_LO(Q0_obj, Q1_obj, lag, **kwargs)
            elif method == 'moments':
                return self.fit_moments(Q0_obj, Q1_obj)
        elif method == 'spectral':
            raise ValueError("""Spectral method not implemented""")
            # return self.fit_spectral(Q_emp[0,:,:], Q_emp[1,:,:])
        else:
            raise ValueError("""Please enter a valid method: 'lyapunov',
                'moments' or 'spectral'.""")


    def fit_from_cov(self, Q0_obj, Q1_obj, lag=1, method='lyapunov', **kwargs):
        """
        Wrapper for model fiting from objective covariance matrices to call the
        specific method. Useful to check the fitting procedure with exact
        theoretical covariances.

        Parameters
        ----------
        Q0_obj : ndarray (2d) of shape (n,n)
            Zero-lag covariance matrix.
        Q1_obj : ndarray (2d) of shape (n,n)
            Lagged covariance matrix.
        lag : int, optional, default=1
            Lag in sampling time steps.
        method : string, optional, default: 'lyapunov'
            Set the optimization method; should be 'lyapunov' or 'moments'.

        Returns
        -------
        J : ndarray (2d) of shape (self.n,self.n)
            The estimated Jacobian.
        Sigma : ndarray (2d) of shape (self.n,self.n)
            Estimated input noise covariance.
        d_fit : dictionary
            A dictionary with diagnostics of the fit (see 'fit' method for
            details).
        """
        if (not type(Q0_obj) == np.ndarray) or (not Q0_obj.ndim == 2) or \
           (not type(Q1_obj) == np.ndarray) or (not Q1_obj.ndim == 2) or \
           (not Q0_obj.shape==Q1_obj.shape):
            raise TypeError("""Argument Q0_obj and Q1_obj must be matrices
                            of shape (nodes x nodes).""")
        self.n = Q0_obj.shape[0]

        # call adequate method for optimization
        if method == 'lyapunov':
            return self.fit_LO(Q0_obj, Q1_obj, lag, **kwargs)
        elif method == 'moments':
            return self.fit_moments(Q0_obj, Q1_obj)
        else:
            raise ValueError("""Please enter a valid method: 'lyapunov',
                'moments'.""")


    def fit_LO(self, Q0_obj, Q1_obj, lag, tau=None, mask_Q=None, mask_C=None,
            mask_Sigma=None, min_C=-1e10, max_C=1e10, min_Sigma_diag=0.0,
            regul_C=0.0, regul_Sigma=0.0, eta_C=0.05, eta_tau=0.0,
            eta_Sigma=0.05, max_iter=500, min_iter=10, algo_version='2021'):
        """
        Estimation of MOU parameters (connectivity C with inverse time constant
        -1/tau on diagonal, input/noise covariance Sigma) performed by Lyapunov
        optimization as in: Gilson et al. Plos Computational Biology (2016);
        Gilson et al. Network Neuroscience (2020); and new version as of 2021.
        For real data, the time constant tau is typically not optimized, but
        calibrated using the (log) autocovariances and set homogeneously to all
        nodes.

        Parameters
        ----------
        Q0_obj : ndarray (2d) of shape (n,n)
            The zero-lag covariance matrix as objective to reproduce
        Q1_obj : ndarray (2d) of shape (n,n)
            The lagged covariance matrix as objective to reproduce
        lag : int
            Lag in real time corresponding to Q1_obj. Default are set in 'fit'
            and 'fit_th' methods.
        tau : scalar, optional, default: None
            The relaxation time-constant of the MOU process.
        mask_Q : boolean ndarray (2d) of shape (self.n,self.n), optional, default: None
            Mask for elements of objective covariance matrices Q0_obj and
            Q1_obj to be taken into account in optimization. By default, all
            elements contribute.
        mask_C : boolean ndarray (2d) of shape (self.n,self.n), optional, default: None
            Mask for tunable elements of connectivity matrix, for example
            estimated by DTI. By default, all connections are allowed (except
            self-connections on diagonal, corresponding to -1/tau)
        mask_Sigma : boolean ndarray (2d) of shape (n,n), optional, default: None
            Mask tunable elements of input covariance matrix Sigma. By default,
            only diagonal elements of Sigma are tuned.
        regul_C : float, optional, default: 0.0
            Regularization parameter for C. First try values in range
            (0.0-0.5).
        regul_Sigma : float, optional, default: 0.0
            Regularization parameter for Sigma. Useful when  fitting
        min_C : float, optional, default: -1e10
            Minimum bound for connectivity estimate C. For instance, useful to
            prevent negative weights (too negative can lead the system to
            dynamic instability), especially for empirical/noisy signals.
        max_C : float, optional, default: 1e10
            Maximum bound for connectivity estimate C. Usually not necessary.
        min_Sigma_diag : float, optional, default: 0.0
            Minimum bound for input variance estimate Sigma (on diagonal).
        eta_C : float, optional, default: 0.05
            Learning rate for connectivity C.
        eta_tau : float, optional, default: 0.0
            Learning rate for tau (inverse on diagonal of Jacobian J).
        eta_Sigma : float, optional, default: 0.05
            Learning rate for Sigma.
        max_iter : int, optional, default=500
            Maximum for optimization iteration steps. If final number of
            iterations reaches this maximum, it means the algorithm has not
            converged (warning raised).
        min_iter : int, optional, default=10
            Minimum for optimization iteration steps before stopping (in case
            of initial increase of model error).
        algo_version : string, optional, default: '2021'
            Version of the optimization update; prefer the latest '2021', but
            for comparison the older versions are 'PCB2016' and 'NeNe2020'.

        Returns
        -------
        J : ndarray (2d) of shape (self.n,self.n)
            The estimated Jacobian (with number of nodesw self.n=n from Q0_obj
            and Q1_obj).
        Sigma : ndarray (2d) of of shape (self.n,self.n)
            Estimated input noise covariance.
        d_fit : dictionary
            A dictionary with diagnostics of the fit. Important keys are:
            'iterations', in addition to 'distance' (model error) and
            'correlation' (goodness of fit) and their history over the
            optimization (see also 'fit' method).
        """
        # TODO: make better graphics (deal with axes separation, etc.)
        if algo_version=='2021':
            def update_J(Delta_Q0, Delta_Q1, Q0, Q1, J):
                return np.dot( np.linalg.pinv(Q0), - Delta_Q0 + np.dot( Delta_Q1, spl.expm(-J) ) )
        elif algo_version=='NeNe2020':
            # Q0^-1
            def update_J(Delta_Q0, Delta_Q1, Q0, Q1, J):
                return np.dot( np.linalg.pinv(Q0), Delta_Q0 ) \
                     + np.dot( Delta_Q0, np.linalg.pinv(Q0) ) \
                     + np.dot( np.linalg.pinv(Q1), Delta_Q1 ) \
                     + np.dot( Delta_Q1, np.linalg.pinv(Q1) )
        elif algo_version=='PCB2016':
            def update_J(Delta_Q0, Delta_Q1, Q0, Q1, J):
                return np.dot( np.linalg.pinv(Q0), Delta_Q0 + np.dot( Delta_Q1, spl.expm(-J) ) )
        else:
            raise ValueError("""Unknown 'algo_version' for LO optimization""")

        # Autocovariance time constant (exponential decay)
        assert Q0_obj.shape==Q1_obj.shape, """Objective covariance matrices
            should have same shape"""
        # if tau==None:
        #     ac = np.hstack((Q0_obj.diagonal(), Q1_obj.diagonal()))
        #     log_ac = np.log( np.maximum( ac, ac.max() * 1e-6 ) )
        #     v_lag = np.hstack(([0]*self.n, [lag]*self.n))
        #     lin_reg = np.polyfit(v_lag, log_ac, 1)
        #     tau_obj = -1.0 / lin_reg[0]
        # else:
        #     #assert type(tau)==float
        #     assert isinstance(tau, (float, np.floating)),
        #     tau_obj = tau
        #     print('tau_obj provided')

        if tau==None:
            ac = np.hstack((Q0_obj.diagonal(), Q1_obj.diagonal()))
            log_ac = np.log( np.maximum( ac, ac.max() * 1e-6 ) )
            v_lag = np.hstack(([0]*self.n, [lag]*self.n))
            lin_reg = np.polyfit(v_lag, log_ac, 1)
            tau_obj = -1.0 / lin_reg[0]
        elif isinstance(tau, (int, np.integer)):
            tau_obj = float(tau)
            print('tau_obj provided')
        elif isinstance(tau, (float, np.floating)):
            tau_obj = tau
            print('tau_obj provided')
        else:
            raise ValueError( "'tau' must be either None or a scalar number." )

        # mask for objective covariances and parameters to tune (C and Sigma)
        mask_diag = np.eye(self.n, dtype=bool)
        if mask_Q is None:
            mask_Q = np.ones([self.n,self.n], dtype=bool)
        mask_notQ = np.logical_not(mask_Q)
        if mask_C is None:
            # tune all possible connections except self-connections on diagonal
            mask_C = np.logical_not(mask_diag)
        if mask_Sigma is None:
            # independent noise (no cross-covariances for Sigma by default)
            mask_Sigma = np.eye(self.n, dtype=bool)

        # coefficients to normalize the model error of Q0 and Q1
        norm_Q0_obj = np.linalg.norm(Q0_obj)
        norm_Q1_obj = np.linalg.norm(Q1_obj)
        # optional scaling for weight update
        # c0 = norm_Q1_obj / ( norm_Q0_obj + norm_Q1_obj )
        # c1 = 1.0 - c0

        # Initialize network as unconnected
        # Connectivity C = 0
        C = np.zeros([self.n, self.n], dtype=np.float64)
        # vector of time constants
        tau = np.copy(tau_obj)
        # Input covariance such that Sigma = -J.T Q0 - Q0 J (Lyapunov eq) with
        # J being diagonal with terms 1/tau_obj
        Sigma = 2 / tau_obj * Q0_obj
        Sigma[np.logical_not(mask_Sigma)] = 0.0

        # Best distance between model and empirical data
        best_dist = 1e10
        best_Pearson = 0.0

        # Arrays to record model parameters and outputs
        # model error = matrix distance between FC matrices
        dist_Q_hist = np.zeros([max_iter], dtype=np.float64)
        # Pearson correlation between model and objective FC matrices
        simil_Q_hist = np.zeros([max_iter], dtype=np.float64)

        # identity matrix
        id_mat = np.eye(self.n, dtype=np.float64)

        # run the optimization process
        stop_opt = False
        i_iter = 0
        while not stop_opt:

            # calculate Jacobian of dynamical system
            J = -id_mat / tau + C

            # Covariances Q0 without lag and Q1 with lag for model
            Q0 = self.calc_Q0_from_param(J, Sigma)
            Q1 = self.calc_Qlag_from_param(Q0, J, lag)

            # difference matrices between model and objectives
            Delta_Q0 = Q0_obj - Q0
            Delta_Q1 = Q1_obj - Q1
            Delta_Q0[mask_notQ] = 0.0
            Delta_Q1[mask_notQ] = 0.0

            # Calculate error between model and empirical data as evaluated by
            # matrix distance (for Q0 and Qlag)
            dist_Q0 = np.linalg.norm(Delta_Q0) / norm_Q0_obj
            dist_Q1 = np.linalg.norm(Delta_Q1) / norm_Q1_obj
            dist_Q_hist[i_iter] = 0.5 * (dist_Q0 + dist_Q1)

            # Calculate similarity between model and empirical data evaluated
            # by the Pearson correlation coefficient (for Q0 and Qlag)
            simil_Q0 = stt.pearsonr( Q0.reshape(-1), Q0_obj.reshape(-1) )[0]
            simil_Q1 = stt.pearsonr( Q1.reshape(-1), Q1_obj.reshape(-1) )[0]
            simil_Q_hist[i_iter] = 0.5 * (simil_Q0  + simil_Q1)

            # Best fit given by best Pearson correlation coefficient
            # for both Q0 and Qlag (better than matrix distance)
            if dist_Q_hist[i_iter] < best_dist:
                best_dist = dist_Q_hist[i_iter]
                best_Pearson = simil_Q_hist[i_iter]
                J_best = np.copy(J)
                Sigma_best = np.copy(Sigma)
            else:
                # wait at least 5 optimization steps before stopping
                stop_opt = i_iter > min_iter

            # Jacobian update with weighted FC updates depending on respective error
            Delta_J = update_J(Delta_Q0, Delta_Q1, Q0, Q1, J)

            # Update effective conectivity matrix (regularization is L2)
            C[mask_C] += eta_C * Delta_J[mask_C]
            C[mask_C] -= eta_C * regul_C * C[mask_C]
            C[mask_C] = np.clip(C[mask_C], min_C, max_C)

            # update tau
            tau += eta_tau * tau**2 * Delta_J.diagonal().mean()

            # Update noise matrix Sigma (regularization is L2)
            Delta_Sigma = - np.dot(J.T, Delta_Q0) - np.dot(Delta_Q0, J)
            Sigma[mask_Sigma] += eta_Sigma * Delta_Sigma[mask_Sigma]
            Sigma[mask_Sigma] -= eta_Sigma * regul_Sigma * Sigma[mask_Sigma]
            Sigma[mask_diag] = np.maximum(Sigma[mask_diag], min_Sigma_diag)

            # optimize tau
            # Delta_tau =

            # Check if max allowed number of iterations have been reached
            if i_iter >= max_iter-1:
                stop_opt = True
                warnings.warn( """Optimization did not converge. Maximum number
                    of iterations arrived.""",
                    category = RuntimeWarning)
            # Check if iteration has finished or still continues
            if stop_opt:
                self.d_fit['iterations'] = i_iter+1
                self.d_fit['distance'] = best_dist
                self.d_fit['correlation'] = best_Pearson
                self.d_fit['distance history'] = dist_Q_hist[:i_iter+1]
                self.d_fit['correlation history'] = simil_Q_hist[:i_iter+1]
            else:
                i_iter += 1

        # Save the results and return
        self.J = J_best # matrix
        self.Sigma = Sigma_best # matrix

        self.d_fit['status'] = 'fitted'

        return self


    def fit_moments(self, Q0_obj, Q1_obj, mask_C=None):
        """
        Estimation of MOU parameters (connectivity C, noise covariance Sigma,
        and time constant tau) with moments method.

        Parameters
        ----------
        X : ndarray (2d) of shape (n,n)
            The zero-lag covariance matrix of the time series to fit.
        mask_C : boolean ndarray (2d) of shape (n,n), optional, default: None
            Mask of known non-zero values for connectivity matrix, for example
            estimated by DTI.

        Returns
        -------
        J : ndarray (2d) of shape (n,n)
            The estimated Jacobian.
        Sigma : ndarray (2d) of shape (n,n)
            Estimated noise covariance.
        d_fit : dictionary
            A dictionary with diagnostics of the fit. Keys are: ['iterations',
            'distance', 'correlation'].
        """
        # Jacobian estimate
        inv_Q0 = np.linalg.inv(Q0_obj)
        J = spl.logm( np.dot(inv_Q0, Q1_obj) )
        # Sigma estimate
        Sigma = - np.dot(J.conjugate(), Q0_obj) - np.dot(Q0_obj, J)

        # masks for existing positions
        mask_diag = np.eye(self.n, dtype=bool)
        if mask_C is None:
            # Allow all possible connections to be tuned except self-connections (on diagonal)
            mask_C = np.logical_not(mask_diag)

        # cast to real matrices
        if np.any(np.iscomplex(J)):
            warnings.warn( "Complex values in J; casting to real!",
                category=RuntimeWarning )
        J_best = np.real(J)
        J_best[np.logical_not(np.logical_or(mask_C,mask_diag))] = 0
        if np.any(np.iscomplex(Sigma)):
            warnings.warn( "Complex values in Sigma; casting to real!",
                category=RuntimeWarning )
        Sigma_best = np.real(Sigma)

        # model theoretical covariances with real J and Sigma
        Q0 = spl.solve_continuous_lyapunov(J_best.T, -Sigma_best)
        Q1 = np.dot( Q0, spl.expm(J_best) )

        # Calculate error between model and empirical data for Q0 and FC_tau (matrix distance)
        dist_Q0 = np.linalg.norm(Q0 - Q0_obj) / np.linalg.norm(Q0_obj)
        dist_Q1 = np.linalg.norm(Q1 - Q1_obj) / np.linalg.norm(Q1_obj)
        self.d_fit['distance'] = 0.5 * (dist_Q0 + dist_Q1)

        # Average correlation between empirical and theoretical
        simil_Q0 = stt.pearsonr( Q0.reshape(-1), Q0_obj.reshape(-1) )[0]
        simil_Q1 = stt.pearsonr( Q1.reshape(-1), Q1_obj.reshape(-1) )[0]
        self.d_fit['correlation'] = 0.5 * (simil_Q0 + simil_Q1)

        # Save the results and return
        self.J = J_best # matrix
        self.Sigma = Sigma_best # matrix

        self.d_fit['status'] = 'fitted'

        return self


    def score(self):
        """
        Returns the correlation between goodness of fit of the MOU to the
        data, measured by the Pearson correlation between the obseved
        covariances and the model covariances.
        """
        try:
            return self.d_fit['correlation']
        except:
            warnings.warn( "The model should be fitted first.", category=RuntimeWarning )
            return np.nan

    def model_covariance(self, tau=0.0):
        """
        Calculates theoretical (lagged) covariances of the model given the
        parameters (forward step). Notice that this is not the empirical
        covariance matrix as estimated from simulated time series.

        Parameters
        ----------
        tau : float, optional, default: 0.0
            The time lag to calculate the covariance. It can be a positive or
            negative.

        Returns
        -------
        FC : ndarray of rank-2
            The (lagged) covariance matrix.
        """

        # Calculate zero lag-covariance Q0 by solving Lyapunov equation
        Q0 = spl.solve_continuous_lyapunov(self.J.T, -self.Sigma)
        # Calculate the effect of the lag (still valid for tau = 0.0)
        if tau >= 0.0:
            return np.dot(Q0, spl.expm(tau * self.J))
        else:
            return np.dot(spl.expm(-tau * self.J.T), Q0)


    def simulate(self, T=100, dt=0.05, sampling_step=1., random_state=None):
        # TODO: change the integration step to be more accurate, with np.exp
        """
        Simulate the MOU process with simple Euler integration defined by the
        time step.

        Parameters
        ----------
        T : int, optional, default = 100
            Duration of the simulation.
        dt : scalar, optional, default: 0.05
            Integration time step.
        sampling_step : float, optional, default: 1.0
            Period for subsampling the generated time series.
        random_state : long or int, optional, default: None
            Description here ...

        Returns
        --------
        ts : ndarray (2d) of shape (T,n)
            Time series of simulated network activity.

        Notes
        -----
        It is possible to include an acitvation function to
        give non linear effect of network input; here assumed to be identity
        """
        # 0) SECURITY CHECKS
        if dt<0.:
            raise ValueError("Integration step has to be positive. dt<0 given.")
        if T<=dt:
            raise ValueError("Duration of simulation too short. T<dt given.")
        if sampling_step<dt:
            raise ValueError("Decrease dt or increase sampling_step. sampling_step<dt given.")
        # set seed for reproducibility
        if random_state is not None:
            np.random.seed(random_state)

        # 1) PREPARE FOR THE SIMULATION
        # 1.1) Simulation time
        # initial period to remove effect of initial conditions
        T0 = int(10. / (-self.J.diagonal()).max())
        # Sampling to get 1 point every second
        n_sampl = int(sampling_step / dt)
        # simulation time in discrete steps
        n_T = int( np.ceil(T / dt) )
        n_T0 = int(T0 / dt)

        # 1.2) Initialise the arrays
        # Array for generated time-series
        ts = np.zeros([int(n_T/n_sampl), self.n], dtype=np.float64)
        # Set initial conditions for activity
        x_tmp = np.random.normal(size=[self.n])
        # Generate noise for all time steps before simulation
        noise = np.random.normal(size=[n_T0+n_T, self.n], scale=(dt**0.5))

        # 2) RUN THE SIMULATION
        # rescaled inputs and Jacobian with time step of simulation
        mu_dt = self.mu * dt
        J_dt = self.J * dt
        # calculate square root matrix of Sigma
        sqrt_Sigma = spl.sqrtm(self.Sigma)
        for t in np.arange(n_T0+n_T):
            # update of activity
            x_tmp += np.dot(x_tmp, J_dt) + mu_dt + np.dot(sqrt_Sigma, noise[t])
            # Discard first n_T0 timepoints
            if t >= n_T0:
                # Subsample timeseries (e.g. to match fMRI time resolution)
                if np.mod(t-n_T0,n_sampl) == 0:
                    # save the result into the array
                    ts[int((t-n_T0)/n_sampl)] = x_tmp

        return ts


    def simulate_time_resolved(self, T=100, T0=10, dt=0.05, sampling_step=0.1, C_profile=1.0, random_state=None):
        """
        Simulate the MOU process with simple Euler integration defined by the
        time step.

        Parameters
        ----------
        T : int, optional, default: 100
            Duration of simulation.
        dt : float, optional, default: 0.05
            Integration time step.
        sampling_step : float, optional, default: 0.1
            Period for subsampling the generated time series.
        C_profile : float or ndarray (1d) of length n, optional, default: 1.0
            Modulation of C over time (same for all connections).
        random_state : long or int, optional, default: None
            Description here ...

        Returns
        --------
        ts : ndarray (2d) of shape (T,n)
            Time series of simulated network activity.

        Notes
        -----
        It is possible to include an acitvation function to
        give non linear effect of network input; here assumed to be identity
        """
        # 0) SECURITY CHECKS
        if dt<0.:
            raise ValueError("Integration step has to be positive. dt<0 given.")
        if T<=dt:
            raise ValueError("Duration of simulation too short. T<dt given.")
        if sampling_step<dt:
            raise ValueError("Decrease dt or increase sampling_step. sampling_step<dt given.")
        # set seed for reproducibility
        if random_state is not None:
            np.random.seed(random_state)

        # 1) PREPARE FOR THE SIMULATION
        # 1.1) Simulation time
        # initial period to remove effect of initial conditions
        T0 = int(10. / (-self.J.diagonal()).max())
        # Sampling to get 1 point every second
        n_sampl = int(sampling_step / dt)
        # simulation time in discrete steps
        n_T = int( np.ceil(T / dt) )
        n_T0 = int(T0 / dt)

        # XXX TO REFINE
        if not type(C_profile)==np.ndarray: #  C_profile==1.0
            # constant profile
            C_profile = np.ones([n_T0+n_T], dtype=np.float64)
        else:
            # Gaussian modulation
            C_profile_T0 = np.zeros([n_T0+n_T], dtype=np.float64)
            C_profile_T0[n_T0:] = C_profile

        # 1.2) Initialise the arrays
        # Array for generated time-series
        ts = np.zeros([int(n_T/n_sampl), self.n], dtype=np.float64)
        # Set initial conditions for activity
        x_tmp = np.random.normal(size=[self.n])
        # Generate noise for all time steps before simulation
        noise = np.random.normal(size=[n_T0+n_T, self.n], scale=(dt**0.5))

        # 2) RUN THE SIMULATION
        # rescaled inputs and Jacobian with time step of simulation
        mu_dt = self.mu * dt
        # XXX TO CLEAN
        diag_J_dt = self.J * dt
        diag_J_dt[np.logical_not(np.eye(self.n, dtype=bool))] = 0.0
        C_dt = self.J * dt
        C_dt[np.eye(self.n, dtype=bool)] = 0
        # calculate square root matrix of Sigma
        sqrt_Sigma = spl.sqrtm(self.Sigma)
        for t in np.arange(n_T0+n_T):
            # update of activity
            x_tmp += np.dot(x_tmp, diag_J_dt + C_dt * C_profile_T0[t]) + mu_dt + np.dot(sqrt_Sigma, noise[t])
            # Discard first n_T0 timepoints
            if t >= n_T0:
                # Subsample timeseries (e.g. to match fMRI time resolution)
                if np.mod(t-n_T0,n_sampl) == 0:
                    # save the result into the array
                    ts[int((t-n_T0)/n_sampl)] = x_tmp

        return ts
