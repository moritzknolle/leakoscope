
import keras, jax
import jax.numpy as jnp

#TODO: Fix this class

class CustomModel(keras.Model):
    """
        Model class that:
            adds a custom train_step method to enable efficient logging of model state metric metrics.
        
    Adapted from: https://keras.io/guides/custom_train_step_in_jax/

    """
    
    def train_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state
        x, y = data

        # Get the gradient function.
        grad_fn = jax.value_and_grad(self.compute_loss_and_updates, has_aux=True)

        # Compute the gradients.
        (loss, (y_pred, non_trainable_variables)), grads = grad_fn(
            trainable_variables,
            non_trainable_variables=non_trainable_variables,
            metrics_variables=metrics_variables,
            x=x,
            y=y,
            sample_weight=None,
            training=True,
        )
        l2_params = jnp.sqrt(
            sum([jnp.sum(p * p) for p in jax.tree_util.tree_leaves(trainable_variables)])
        )
        l2_grads = jnp.sqrt(
            sum([jnp.sum(g * g) for g in jax.tree_util.tree_leaves(grads)])
        )
        # Update trainable variables and optimizer variables.
        (
            trainable_variables,
            optimizer_variables,
        ) = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )
        l2_updates = jnp.sqrt(
            sum(
                [
                    jnp.sum(u * u)
                    for u in jax.tree_util.tree_leaves(optimizer_variables)
                ]
            )
        )
        # Update metrics.
        new_metrics_vars = []
        logs = {"l2_updates": l2_updates, "l2_grads": l2_grads, "l2_params": l2_params}
        for metric in self.metrics:
            this_metric_vars = metrics_variables[
                len(new_metrics_vars) : len(new_metrics_vars) + len(metric.variables)
            ]
            if metric.name == "loss":
                this_metric_vars = metric.stateless_update_state(this_metric_vars, loss)
            else:
                this_metric_vars = metric.stateless_update_state(
                    this_metric_vars, y, y_pred
                )
            logs[metric.name] = metric.stateless_result(this_metric_vars)
            new_metrics_vars += this_metric_vars

        # Return metric logs and updated state variables.
        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            new_metrics_vars,
        )
        return logs, state


        def train_step(self, state, data):
            (
                trainable_variables,
                non_trainable_variables,
                optimizer_variables,
                metrics_variables,
            ) = state
            x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
            grad_fn = jax.value_and_grad(
                self.compute_loss_and_updates, has_aux=True
            )
            (loss, aux), grads = grad_fn(
                trainable_variables,
                non_trainable_variables,
                metrics_variables,
                x,
                y,
                sample_weight,
                training=True,
                optimizer_variables=optimizer_variables,
            )
            (unscaled_loss, y_pred, non_trainable_variables, metrics_variables) = (
                aux
            )

            (
                trainable_variables,
                optimizer_variables,
            ) = self.optimizer.stateless_apply(
                optimizer_variables, grads, trainable_variables
            )
            # calculate l2 norm of weights, gradients and updates
            l2_params = jnp.sqrt(
            sum([jnp.sum(p * p) for p in jax.tree_util.tree_leaves(trainable_variables)])
            )
            l2_grads = jnp.sqrt(
                sum([jnp.sum(g * g) for g in jax.tree_util.tree_leaves(grads)])
            )
            l2_updates = jnp.sqrt(
                sum(
                    [
                        jnp.sum(u * u)
                        for u in jax.tree_util.tree_leaves(optimizer_variables)
                    ]
                )
            )
            with backend.StatelessScope(
                state_mapping=[
                    (ref_v, v)
                    for ref_v, v in zip(self.metrics_variables, metrics_variables)
                ]
            ) as scope:
                self._loss_tracker.update_state(
                    unscaled_loss, sample_weight=tree.flatten(x)[0].shape[0]
                )
                logs = self.compute_metrics(x, y, y_pred, sample_weight)

            logs = logs | {"l2_params": l2_params, "l2_grads": l2_grads, "l2_updates": l2_updates}

            new_metrics_variables = []
            for ref_v in self.metrics_variables:
                new_v = scope.get_current_value(ref_v)
                if new_v is None:
                    new_v = ref_v.value
                new_metrics_variables.append(new_v)
            metrics_variables = new_metrics_variables

            state = self._enforce_jax_state_sharding(
                trainable_variables,
                non_trainable_variables,
                optimizer_variables,
                metrics_variables,
            )
            return logs, state