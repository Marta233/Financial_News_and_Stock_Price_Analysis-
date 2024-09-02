import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

class CorrelationAnalysis:
    def __init__(self, df):
        self.df = df