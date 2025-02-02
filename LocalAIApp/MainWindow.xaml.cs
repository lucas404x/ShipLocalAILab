using LLama;
using LLama.Abstractions;
using LLama.Common;
using LLama.Sampling;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using Microsoft.Win32;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Input;

namespace LocalAIApp;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        DataContext = new MainWindowVM();
    }
}

public class MainWindowVM : INotifyPropertyChanged
{
    #region NotifyPropertyChanged

    public event PropertyChangedEventHandler? PropertyChanged;

    #endregion

    #region Properties

    private readonly LLMModel _model = new();

    private string _modelPath = string.Empty;
    public string ModelPath
    {
        get => _modelPath;
        set
        {
            _modelPath = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(EnableAskButton));
        }
    }

    private bool _isProcessing = false;
    public bool IsProcessing
    {
        get => _isProcessing;
        set
        {
            _isProcessing = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(EnableLoadModelButton));
            OnPropertyChanged(nameof(EnableAskButton));
            OnPropertyChanged(nameof(CopyButtonVisibility));
        }
    }

    public bool EnableLoadModelButton => !_isProcessing;

    public bool EnableAskButton 
        => !IsProcessing && !string.IsNullOrWhiteSpace(ModelPath) && !string.IsNullOrWhiteSpace(QuestionBox);

    public Visibility CopyButtonVisibility => IsProcessing || string.IsNullOrWhiteSpace(AnswerBox) ? Visibility.Collapsed : Visibility.Visible;

    private string _questionBox = string.Empty;
    public string QuestionBox
    {
        get => _questionBox;
        set
        {
            _questionBox = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(EnableAskButton));
        }
    }

    private string _answerBox = string.Empty;
    public string AnswerBox
    {
        get => _answerBox;
        set
        {
            _answerBox = value;
            OnPropertyChanged();
        }
    }

    #endregion

    #region Commands

    public ICommand LoadModelFile { get; private set; }
    public ICommand AskCommand { get; private set; }
    public ICommand CopyOutputCommand { get; private set; }

    #endregion

    public MainWindowVM()
    {
        LoadModelFile = new RelayCommand(LoadModel);
        AskCommand = new RelayCommand(Ask);
        CopyOutputCommand = new RelayCommand(CopyOutput);
    }

    private void LoadModel(object? obj)
    {
        var openFileDialog = new OpenFileDialog
        {
            Filter = "GGUF files (*.gguf)|*.gguf|All files (*.*)|*.*",
            AddExtension = true,
        };

        if (openFileDialog.ShowDialog() == true)
        {
            ModelPath = openFileDialog.FileName;
        }
    }

    private async void Ask(object? obj)
    {
        AnswerBox = string.Empty;
        try
        {
            IsProcessing = true;

            await _model.LoadModel(ModelPath);

            await foreach (var answer in _model.Chat(QuestionBox))
            {
                AnswerBox += answer;
            }
        }
        catch (Exception e)
        {
            AnswerBox = e.Message;
        }
        finally
        {
            IsProcessing = false;
        }
    }

    private void CopyOutput(object? obj)
    {
        Clipboard.SetText(AnswerBox);
    }


    protected void OnPropertyChanged([CallerMemberName] string p = "")
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(p));
    }
}

public class LLMModel : IDisposable
{
    private string? _modelPath;

    private LLamaWeights? _model;
    private StatelessExecutor? _executor;

    public async Task LoadModel(string modelPath)
    {
        if (_modelPath == modelPath) return;
        Dispose();

        _modelPath = modelPath;

        var parameters = new ModelParams(_modelPath)
        {
            GpuLayerCount = 11, // How many layers to offload to GPU. Please adjust it according to your GPU memory.
            ContextSize = 4096
        };

        _model = await LLamaWeights.LoadFromFileAsync(parameters);
        _executor = new StatelessExecutor(_model, parameters)
        {
            ApplyTemplate = true,
            SystemMessage = @""
        };
    }

    public IAsyncEnumerable<string> Chat(string prompt)
    {
        var inferenceParams = new InferenceParams()
        {
            MaxTokens = 2048,
            SamplingPipeline = new DefaultSamplingPipeline
            {
                Temperature = 0.4f
            }
        };
        return _executor!.InferAsync(prompt, inferenceParams);
    }

    public async Task<string> GetResponse(string prompt)
    {
        var chat = _executor!.AsChatClient();
        var response = await chat.CompleteAsync([new ChatMessage(ChatRole.User, prompt)]);
        return response.Choices[0].Text ?? string.Empty;
    }

    public void Dispose()
    {
        _executor = null;
        _model?.Dispose();
    }
}