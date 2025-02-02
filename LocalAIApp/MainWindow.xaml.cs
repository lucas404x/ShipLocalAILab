using LLama;
using LLama.Abstractions;
using LLama.Common;
using LLama.Native;
using LLama.Sampling;
using Microsoft.Extensions.AI;
using Microsoft.Win32;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Input;
using System.Diagnostics;

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

        NativeLibraryConfig.All
#if DEBUG
            .WithLogCallback((level, message) => Debug.WriteLine($"{level}: {message}"))
#endif
            .WithCuda();
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

    private int _gpuLayerCount = 14;
    public int GPULayerCount
    {
        get => _gpuLayerCount;
        set
        {
            _gpuLayerCount = value;
            OnPropertyChanged();
        }
    }

    private uint _contextSize = 2048;
    public uint ContextSize
    {
        get => _contextSize;
        set
        {
            _contextSize = value;
            OnPropertyChanged();
        }
    }

    private bool _applyTemplate = true;
    public bool ApplyTemplate
    {
        get => _applyTemplate;
        set
        {
            _applyTemplate = value;
            OnPropertyChanged();
        }
    }

    private string _systemPrompt = "You're a prompt compressor specialist. Your objective is get the prompt provided by the user and making it small but keeping its intent. The prompt will be used for another LLM, so make explicit the \"you're\". Just return the compressed prompt, nothing more around it.";
    public string SystemPrompt
    {
        get => _systemPrompt;
        set
        {
            _systemPrompt = value;
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
            OnPropertyChanged(nameof(EnableButtons));
            OnPropertyChanged(nameof(EnableAskButton));
            OnPropertyChanged(nameof(CopyButtonVisibility));
        }
    }

    public bool EnableButtons => !_isProcessing;

    public bool EnableAskButton 
        => EnableButtons 
        && !string.IsNullOrWhiteSpace(ModelPath) 
        && !string.IsNullOrWhiteSpace(SystemPrompt) 
        && !string.IsNullOrWhiteSpace(QuestionBox);

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

            await _model.LoadModel(new LLMModelConfig(ModelPath, SystemPrompt, GPULayerCount, ContextSize, ApplyTemplate));

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
        => Clipboard.SetText(AnswerBox);

    protected void OnPropertyChanged([CallerMemberName] string p = "")
        => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(p));
}

public record LLMModelConfig(
    string ModelPath, 
    string SystemPrompt,
    int GPULayerCount,
    uint ContextSize,
    bool ApplyTemplate);

public class LLMModel : IDisposable
{
    private LLMModelConfig _modelConfig = null!;
    private LLamaWeights _model = null!;
    private StatelessExecutor? _executor;

    public async Task LoadModel(LLMModelConfig modelConfig)
    {
        if (_modelConfig == modelConfig) return;
        Dispose();

        var sw = Stopwatch.StartNew();

        _modelConfig = modelConfig;

        var parameters = await Task.Run(() => // when load large models with cuda the operation may take a while, hanging the application for some seconds
        {
            return new ModelParams(_modelConfig.ModelPath)
            {
                GpuLayerCount = _modelConfig.GPULayerCount, // How many layers to offload to GPU. Please adjust it according to your GPU memory.
                ContextSize = _modelConfig.ContextSize,
            };
        });

        Debug.WriteLine($"Parameters loaded: {sw.ElapsedMilliseconds}ms");

        _model = await LLamaWeights.LoadFromFileAsync(parameters);

        Debug.WriteLine($"Model loaded: {sw.ElapsedMilliseconds}ms");

        _executor = new StatelessExecutor(_model, parameters)
        {
            ApplyTemplate = _modelConfig.ApplyTemplate,
            SystemMessage = _modelConfig.ApplyTemplate ? _modelConfig.SystemPrompt : null,
        };

        Debug.WriteLine($"Executor loaded: {sw.ElapsedMilliseconds}ms");
    }

    public IAsyncEnumerable<string> Chat(string prompt)
    {
        var inferenceParams = new InferenceParams()
        {
            MaxTokens = 1024,
            SamplingPipeline = new DefaultSamplingPipeline
            {
                Temperature = 0.4f
            }
        };
        if (!_modelConfig.ApplyTemplate)
        {
            prompt = $"{_modelConfig.SystemPrompt}\n{prompt}";
        }
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

        GC.SuppressFinalize(this);
    }
}