import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np

# Define the CNOT gate matrix
cnot_matrix = np.array([
    [1, 0, 0, 0],  # |00⟩ -> |00⟩
    [0, 1, 0, 0],  # |01⟩ -> |01⟩
    [0, 0, 0, 1],  # |10⟩ -> |11⟩
    [0, 0, 1, 0]   # |11⟩ -> |10⟩
])

# Basis states
basis_states = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.H1("CNOT Gate Interactive Visualizer"),
    dcc.Slider(
        id="state-slider",
        min=0,
        max=3,
        step=1,
        value=0,
        marks={i: state for i, state in enumerate(basis_states)}
    ),
    dcc.Graph(id="dual-column-graph"),
    dcc.Markdown(id="description", style={"marginTop": "20px", "fontSize": "18px"})
])

# Define the callback for the interactive graph and description
@app.callback(
    [Output("dual-column-graph", "figure"),
     Output("description", "children")],
    [Input("state-slider", "value")]
)
def update_dual_column(selected_index):
    # Prepare input and output states
    input_vector = np.zeros(4)
    input_vector[selected_index] = 1
    output_vector = np.dot(cnot_matrix, input_vector)

    # Create bar chart for dual-column visualization
    fig = go.Figure()

    # Input column
    fig.add_trace(go.Bar(
        x=basis_states,
        y=input_vector,
        name="Input States",
        marker=dict(color="blue"),
        xaxis="x1",
        yaxis="y1"
    ))

    # Output column
    fig.add_trace(go.Bar(
        x=basis_states,
        y=output_vector,
        name="Output States",
        marker=dict(color="orange"),
        xaxis="x2",
        yaxis="y1"
    ))

    fig.update_layout(
        title=f"Transition: {basis_states[selected_index]} → {basis_states[np.argmax(output_vector)]}",
        xaxis=dict(domain=[0, 0.45], title="Input States"),
        xaxis2=dict(domain=[0.55, 1], title="Output States"),
        yaxis=dict(range=[0, 1], title="Probability"),
        barmode="group",
        showlegend=True
    )

    # Generate description
    if np.count_nonzero(output_vector) == 1:  # Single deterministic state
        output_state = basis_states[np.argmax(output_vector)]
        description = (f"**Input state:** {basis_states[selected_index]}\n\n"
                       f"**Output state:** {output_state}\n\n"
                       f"The CNOT gate flipped the target qubit if the control qubit was |1⟩.")
    else:  # For superpositions or mixed states (not applicable here but for completeness)
        output_description = ", ".join([
            f"{basis_states[i]} ({output_vector[i]:.2f})"
            for i in range(4) if output_vector[i] > 0
        ])
        description = (f"**Input state:** {basis_states[selected_index]}\n\n"
                       f"**Output states:** {output_description}\n\n"
                       f"The CNOT gate creates these probabilities based on the input state.")
    
    return fig, description

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
