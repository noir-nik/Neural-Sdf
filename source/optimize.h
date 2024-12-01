#pragma once

#include <vector>
#include <cstdio>
#include <cmath>

enum class OptimizerType {
    Adam,
    BasicGD
};

template<typename ParamType, OptimizerType Type>
class Optimizer {
public:
    std::vector<ParamType> m_gradient; // Gradient
private:
    std::vector<ParamType> m_momentum; // First moment vector
    std::vector<ParamType> m_velocity; // Second moment vector
    ParamType m_alpha; // Learning rate
    ParamType m_beta1; // Decay rate for the first moment estimates
    ParamType m_beta2; // Decay rate for the second moment estimates
    ParamType m_epsilon; // Small value to prevent division by zero
    int step = 0; // Time step
    size_t grad_size = 0;
	size_t m_size = 0;

public:
    Optimizer(ParamType alpha, ParamType beta1 = 0.5, ParamType beta2 = 0.999, ParamType epsilon = 1e-8)
        : m_alpha(alpha), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon), step(0) {}

	void setSize (size_t num_params_T, size_t gradient_size = 0) {
		m_size = num_params_T;
        if (gradient_size < num_params_T){
            grad_size = num_params_T;
        } else {
            grad_size = gradient_size;
        }
        m_gradient.clear();
		m_gradient.resize(grad_size, ParamType(0));
        if constexpr (Type == OptimizerType::Adam) {
            m_momentum.clear();
            m_momentum.resize(m_size, ParamType(0));
            m_velocity.clear();
            m_velocity.resize(m_size, ParamType(0));
        }
	}

	void set_grad(ParamType val){
		std::fill(m_gradient.begin(), m_gradient.end(), val);
	}

	void zero_grad(){
		std::fill(m_gradient.begin(), m_gradient.end(), ParamType(0));
	}

	std::vector<ParamType>& gradient() {
		return m_gradient;
	}

	ParamType* ptr_grad() {
		return &m_gradient[0];
	}

    void update(void* params_ptr, bool use_momentum, ParamType M_basic_step = 0) {
        updateImpl(params_ptr, use_momentum, M_basic_step);
    }

    void update(void* params_ptr) {
        updateImpl(params_ptr);
    }

    ParamType get_learning_rate() {
        return m_alpha;
    }

    void set_learning_rate(ParamType alpha) {
        m_alpha = alpha;
    }


private:
    // Method to update the optimizer with a new gradient
    inline void updateImpl(void* params_ptr, bool use_momentum = true, ParamType M_basic_step = 0) {
        
		ParamType *params= static_cast<ParamType*>(params_ptr);

        if constexpr (Type == OptimizerType::Adam) {
            if (use_momentum) {
                updateMomentumImpl(params_ptr);
            } else {
                updateBasicImpl(params_ptr, M_basic_step);
            }
        } else if constexpr (Type == OptimizerType::BasicGD) {
            ParamType T_learning_rate;
            if (M_basic_step == 0) {
                T_learning_rate = m_alpha;
            } else {
                T_learning_rate = M_basic_step;
            }
            updateBasicImpl(params_ptr, T_learning_rate);
        }
    }

    inline void updateBasicImpl(void* params_ptr, ParamType M_basic_step) {
        ParamType *params= static_cast<ParamType*>(params_ptr);
        for (size_t i = 0; i < m_size; ++i) {
            params[i] -= M_basic_step * m_gradient[i];
        }
    }

    inline void updateMomentumImpl(void* params_ptr) {

        // Increment time step
        step++;
        ParamType *params= static_cast<ParamType*>(params_ptr);
        for (size_t i = 0; i < m_size; ++i) {
            m_momentum[i] = m_beta1 * m_momentum[i] + (1 - m_beta1) * m_gradient[i];
            m_velocity[i] = m_beta2 * m_velocity[i] + (1 - m_beta2) * m_gradient[i] * m_gradient[i];

            ParamType alpha_t = m_alpha * std::sqrt(1 - std::pow(m_beta2, step)) / (1 - std::pow(m_beta1, step));
            params[i] -= alpha_t * m_momentum[i] / (std::sqrt(m_velocity[i]) + m_epsilon);
        }
    }
};
