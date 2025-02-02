using System.Windows.Input;

namespace LocalAIApp
{
    public class RelayCommand(Action<object?> executeAction) : ICommand
    {
        private readonly Action<object?> execute = executeAction;

        public event EventHandler? CanExecuteChanged;

        public bool CanExecute(object? parameter) => true;

        public void Execute(object? parameter) => execute(parameter);
    }
}
